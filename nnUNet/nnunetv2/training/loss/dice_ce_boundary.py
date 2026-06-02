# nnunetv2/training/loss/dc_ce_boundary.py

from __future__ import annotations

from typing import Optional, Sequence, Union, List
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]


def _binary_morphological_boundary(mask_01: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    mask_01: (B,1,H,W) float/bool in {0,1}
    returns: (B,1,H,W) bool boundary band via morphological gradient (dilate - erode)
    """
    if k < 3 or k % 2 == 0:
        raise ValueError(f"boundary_kernel must be odd and >=3, got {k}")

    mask_01 = mask_01.float()
    dil = F.max_pool2d(mask_01, kernel_size=k, stride=1, padding=k // 2)
    ero = 1.0 - F.max_pool2d(1.0 - mask_01, kernel_size=k, stride=1, padding=k // 2)
    boundary = (dil - ero).clamp(0, 1)
    return (boundary > 0)


def _default_ds_weights(n: int) -> List[float]:
    # 1, 1/2, 1/4, ... normalized
    w = [1.0 / (2 ** i) for i in range(n)]
    s = float(sum(w))
    return [float(x) / s for x in w]




class DC_CE_Boundary_loss(nn.Module):
    def __init__(self, soft_dice_kwargs,ce_kwargs, weight_ce = 1.0,weight_dice = 1.0,
                weight_boundary = 0.5,boundary_boost = 4.0, boundary_kernel = 3, ignore_label = None,
                dice_class=SoftDiceLoss,  peri_only = True, peri_label_id = 2, deep_supervision_weights = None):
        super(DC_CE_Boundary_loss, self).__init__()

        # ---- ignore label handling consistent with nnU-Net ----
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_boundary = float(weight_boundary)

        self.boundary_boost = float(boundary_boost)
        self.boundary_kernel = int(boundary_kernel)

        self.peri_only = bool(peri_only)
        self.peri_label_id = int(peri_label_id)

        self.ds_weights = list(deep_supervision_weights) if deep_supervision_weights is not None else None

        # ---- nnU-Net components ----
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None


        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # ---- Boundary-weighted CE ----
        # Build boundary mask from GT (not prediction)
        tgt_lbl = target[:, 0].long()  # (B,H,W)

        if self.peri_only:
            region = (tgt_lbl == self.peri_label_id).unsqueeze(1)  # (B,1,H,W) bool
        else:
            region = (tgt_lbl != 0).unsqueeze(1)  # foreground union

        boundary = _binary_morphological_boundary(region, k=self.boundary_kernel)  # (B,1,H,W) bool

        # per-pixel CE (respect ignore label)
        ignore_index = self.ignore_label if self.ignore_label is not None else -100
        ce_px = F.cross_entropy(net_output, tgt_lbl, reduction="none", ignore_index=ignore_index)  # (B,H,W)

        weight_map = 1.0 + self.boundary_boost * boundary[:, 0].float()  # (B,H,W)

        if self.ignore_label is not None:
            valid = (tgt_lbl != self.ignore_label)
            bnd_ce = (ce_px * weight_map)[valid].mean() if valid.any() else ce_px.mean() * 0.0
        else:
            bnd_ce = (ce_px * weight_map).mean()

        # ---- Combine ----
        return self.weight_dice * dc_loss + self.weight_ce * ce_loss + self.weight_boundary * bnd_ce
    


