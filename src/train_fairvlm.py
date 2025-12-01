import os
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.models.fairvlm import FairVLM




class FairSegDataset(Dataset):
    """
    Generic dataset for Harvard-FairSeg-style CSV.

    Expected CSV columns (you can adapt):
      - image_path: relative or absolute path to image
      - mask_path:  relative or absolute path to segmentation mask
      - prompt:     textual prompt string
      - demo_*:     demographic columns (e.g., sex, race, ethnicity, language) as ints {0,1}
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        mask_root: str,
        demographic_cols: List[str],
        image_size: int = 512,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.mask_root = mask_root
        self.demographic_cols = demographic_cols
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        full_path = path if os.path.isabs(path) else os.path.join(self.img_root, path)
        img = Image.open(full_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(self.image_size, self.image_size, 3)
             .permute(2, 0, 1)
             .float()) / 255.0
        )
        return img

    def _load_mask(self, path):
        full_path = path if os.path.isabs(path) else os.path.join(self.mask_root, path)
        mask = Image.open(full_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size))
        mask = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
            .view(self.image_size, self.image_size)
            .float()
        )
        mask = (mask > 0).float()  # binary mask
        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = self._load_image(row["image_path"])
        mask = self._load_mask(row["mask_path"])

        prompt = row["prompt"]

        demo_vals = row[self.demographic_cols].values.astype("float32")
        demo = torch.tensor(demo_vals, dtype=torch.float32)

        return img, mask, prompt, demo




def build_backbone(backbone_name: str):
    """
    You should implement SAMedBackbone and LViTBackbone classes
    with methods:
      - encode_image(image)
      - encode_text(prompts)
      - segment(image_embed, text_embeds)
      - text_dim (int attribute)

    and import them here.
    """
    backbone_name = backbone_name.lower()
    if backbone_name == "samed":
        from src.backbones.samed_backbone import SAMedBackbone
        return SAMedBackbone()
    elif backbone_name == "lvit":
        from src.backbones.lvit_backbone import LViTBackbone
        return LViTBackbone()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")




def dice_iou(pred_logits, gt):
    """
    pred_logits: Bx1xHxW (raw logits)
    gt:          Bx1xHxW (0/1)
    """
    pred = torch.sigmoid(pred_logits)
    pred_bin = (pred > 0.5).float()

    smooth = 1e-6
    intersection = (pred_bin * gt).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])
    dice = (2 * intersection + smooth) / (union + smooth)

    inter_iou = (pred_bin * gt).sum(dim=[1, 2, 3])
    union_iou = (pred_bin + gt - pred_bin * gt).sum(dim=[1, 2, 3])
    iou = (inter_iou + smooth) / (union_iou + smooth)

    return dice.mean().item(), iou.mean().item()




def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    log_interval=50,
):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", ncols=100)
    for batch_idx, (img, mask, prompt, demo) in enumerate(pbar):
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)  # Bx1xHxW
        demo = demo.to(device)

        optimizer.zero_grad()
        pred, loss = model(img, prompt, demo, mask_gt=mask)
        loss.backward()
        optimizer.step()

        dice, iou = dice_iou(pred.detach(), mask)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": running_loss / n_batches,
                    "dice": running_dice / n_batches,
                    "iou": running_iou / n_batches,
                }
            )

    return (
        running_loss / max(1, n_batches),
        running_dice / max(1, n_batches),
        running_iou / max(1, n_batches),
    )


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [val]", ncols=100)
    for img, mask, prompt, demo in pbar:
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        demo = demo.to(device)

        pred, loss = model(img, prompt, demo, mask_gt=mask)
        dice, iou = dice_iou(pred, mask)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        n_batches += 1

        pbar.set_postfix(
            {
                "loss": running_loss / n_batches,
                "dice": running_dice / n_batches,
                "iou": running_iou / n_batches,
            }
        )

    return (
        running_loss / max(1, n_batches),
        running_dice / max(1, n_batches),
        running_iou / max(1, n_batches),
    )




def parse_args():
    parser = argparse.ArgumentParser(description="Train FairVLM")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)

    parser.add_argument(
        "--demographic_cols",
        type=str,
        nargs="+",
        default=["sex", "race", "ethnicity", "language"],
        help="Column names used to build demographic multi-hot vectors",
    )

    parser.add_argument("--backbone", type=str, default="samed", choices=["samed", "lvit"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_srcp", action="store_true", default=True)
    parser.add_argument("--use_dafn", action="store_true", default=True)
    parser.add_argument("--use_fcl", action="store_true", default=True)

    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--image_size", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset / DataLoader
    train_ds = FairSegDataset(
        csv_path=args.train_csv,
        img_root=args.img_root,
        mask_root=args.mask_root,
        demographic_cols=args.demographic_cols,
        image_size=args.image_size,
    )
    val_ds = FairSegDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        mask_root=args.mask_root,
        demographic_cols=args.demographic_cols,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Backbone + FairVLM
    backbone = build_backbone(args.backbone)
    model = FairVLM(
        backbone=backbone,
        use_srcp=args.use_srcp,
        use_dafn=args.use_dafn,
        use_fcl=args.use_fcl,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_dice = 0.0
    best_path = os.path.join(args.output_dir, f"fairvlm_{args.backbone}_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_dice, val_iou = validate(model, val_loader, device, epoch)

        print(
            f"[Epoch {epoch}] "
            f"Train: loss={train_loss:.4f}, dice={train_dice:.4f}, iou={train_iou:.4f} | "
            f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                best_path,
            )
            print(f"✅ Saved new best model to {best_path} (val dice={best_val_dice:.4f})")


if __name__ == "__main__":
    main()
