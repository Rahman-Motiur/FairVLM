import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.models.fairvlm import FairVLM
from src.train_fairvlm import FairSegDataset, build_backbone, dice_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FairVLM")

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)

    parser.add_argument(
        "--demographic_cols",
        type=str,
        nargs="+",
        default=["sex", "race", "ethnicity", "language"],
    )

    parser.add_argument("--backbone", type=str, default="samed", choices=["samed", "lvit"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--use_srcp", action="store_true", default=True)
    parser.add_argument("--use_dafn", action="store_true", default=True)
    parser.add_argument("--use_fcl", action="store_true", default=True)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
    for img, mask, prompt, demo in pbar:
        img = img.to(device)
        mask = mask.unsqueeze(1).to(device)
        demo = demo.to(device)

        # For evaluation we don’t need FCL loss, only predictions
        pred, _ = model(img, prompt, demo, mask_gt=mask)
        dice, iou = dice_iou(pred, mask)

        running_dice += dice
        running_iou += iou
        n_batches += 1

        pbar.set_postfix(
            {
                "dice": running_dice / n_batches,
                "iou": running_iou / n_batches,
            }
        )

    avg_dice = running_dice / max(1, n_batches)
    avg_iou = running_iou / max(1, n_batches)

    return avg_dice, avg_iou


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    test_ds = FairSegDataset(
        csv_path=args.test_csv,
        img_root=args.img_root,
        mask_root=args.mask_root,
        demographic_cols=args.demographic_cols,
        image_size=args.image_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Backbone + model
    backbone = build_backbone(args.backbone)
    model = FairVLM(
        backbone=backbone,
        use_srcp=args.use_srcp,
        use_dafn=args.use_dafn,
        use_fcl=args.use_fcl,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Evaluate
    avg_dice, avg_iou = evaluate(model, test_loader, device)
    print(f"\nTest Dice: {avg_dice:.4f} | Test IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
