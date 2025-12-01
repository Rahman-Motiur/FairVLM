import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.srcp import SRCP
from src.modules.dafn import DAFN
from src.modules.fcl import FairnessCalibratedLoss


class FairVLM(nn.Module):
    """
    FairVLM: Fairness + Prompt-Invariant Vision-Language Segmentation Model.
    Implements SRCP, DAFN, and FCL on top of a backbone VLM (e.g., SAMed, LViT).
    """

    def __init__(self, backbone, use_srcp=True, use_dafn=True, use_fcl=True):
        super().__init__()

        self.backbone = backbone
        self.use_srcp = use_srcp
        self.use_dafn = use_dafn
        self.use_fcl = use_fcl

        if use_srcp:
            self.srcp = SRCP()

        if use_dafn:
            self.dafn = DAFN(embed_dim=backbone.text_dim)

        if use_fcl:
            self.fcl = FairnessCalibratedLoss()

    def forward(self, image, prompt, demographic, mask_gt=None):
        """
        image          : BxCxHxW image tensor
        prompt         : list of prompt strings
        demographic    : BxK multi-hot demographic vectors (sex, race, ethnicity, language)
        mask_gt        : ground-truth for segmentation (optional)
        """

        # 1. SRCP – multiple prompt variations (k=3 selected)
        if self.use_srcp:
            counter_prompts = self.srcp.generate(prompt)
        else:
            counter_prompts = prompt

        # 2. Encode text prompts
        text_embeds = self.backbone.encode_text(counter_prompts)

        # 3. DAFN – demographic-aware feature normalization
        if self.use_dafn:
            text_embeds = self.dafn(text_embeds, demographic)

        # 4. Vision encoding
        image_embed = self.backbone.encode_image(image)

        # 5. Segmentation head
        pred_mask = self.backbone.segment(image_embed, text_embeds)

        # 6. Apply FCL loss if GT mask provided
        if mask_gt is not None and self.use_fcl:
            loss = self.fcl(pred_mask, mask_gt, demographic)
            return pred_mask, loss

        return pred_mask
