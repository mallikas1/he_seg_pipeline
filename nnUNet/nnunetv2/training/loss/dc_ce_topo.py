# nnunetv2/training/loss/topo.py

'''
1. label 2 always encloses label 1 
2: label 3 encloses both label 1 and 2. 
due to histology collection process 1 can touch 0 + 1 cannot touch 3 + 2 can touch 0
'''

from __future__ import annotations

from typing import Optional, Sequence, Union, List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class TI_Loss_2D_Adjacency(nn.Module):
    """
    2D topology / anatomy-aware loss based on forbidden label adjacencies.

    This is better than strict inclusion for your case because:
      - label 1 can touch 0
      - label 2 can touch 0
      - but label 1 must NOT touch label 3

    Loss is computed only on pixels involved in forbidden contacts.
    """

    def __init__(
        self,
        connectivity=8,
        forbidden_pairs=((1, 3),),   # symmetric: 1 cannot touch 3
        apply_nonlin=None,
    ):
        super().__init__()

        assert connectivity in (4, 8), "For 2D, connectivity must be 4 or 8"
        self.connectivity = connectivity
        self.forbidden_pairs = [tuple(p) for p in forbidden_pairs]
        self.apply_nonlin = apply_nonlin if apply_nonlin is not None else (lambda x: F.softmax(x, dim=1))
        self.ce_loss_func = nn.CrossEntropyLoss(reduction="none")

        self.register_buffer("kernel", self._make_kernel(connectivity))

    def _make_kernel(self, connectivity):
        if connectivity == 4:
            k = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        else:  # 8-connectivity
            k = np.ones((3, 3), dtype=np.float32)

        k = torch.from_numpy(k)[None, None, :, :]   # (1,1,3,3)
        return k
    
    def _get_mask(self, P, label):
        return (P == label).to(dtype=P.dtype if P.is_floating_point() else torch.float32, device=P.device)
    
    def _has_neighbor(self, mask):
        # mask: (B,1,H,W)
        kernel = self.kernel.to(device=mask.device, dtype=mask.dtype)
        neigh = F.conv2d(mask, kernel, padding=1)
        return (neigh >= 1).to(mask.dtype)

    def critical_pixels_from_prediction(self, P):
        """
        Returns a binary map of pixels involved in forbidden adjacencies.
        P: predicted hard labels, shape (B,1,H,W)
        """
        critical = torch.zeros_like(P, dtype=torch.float32)

        for a, b in self.forbidden_pairs:
            mask_a = self._get_mask(P, a)
            mask_b = self._get_mask(P, b)

            neigh_a = self._has_neighbor(mask_a)
            neigh_b = self._has_neighbor(mask_b)

            # a-pixels touching b, and b-pixels touching a
            violating_a = mask_a * neigh_b
            violating_b = mask_b * neigh_a

            critical = torch.maximum(critical, violating_a)
            critical = torch.maximum(critical, violating_b)

        return critical

    def forward(self, x, y):
        """
        x: logits, shape (B,C,H,W)
        y: GT, shape (B,1,H,W)
        """
        # hard prediction
        x_soft = self.apply_nonlin(x)
        P = torch.argmax(x_soft, dim=1, keepdim=True)  # (B,1,H,W)

        # find critical pixels from current prediction
        critical = self.critical_pixels_from_prediction(P).squeeze(1)  # (B,H,W)

        # pixelwise CE, only on critical pixels
        ce_map = self.ce_loss_func(x, y[:, 0].long())  # (B,H,W)

        masked_ce = ce_map * critical
        denom = torch.clamp(critical.sum(dim=(1, 2)), min=1.0)

        loss = (masked_ce.sum(dim=(1, 2)) / denom).mean()
        return loss


class DC_CE_TI_loss_2D(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1.0,
        weight_dice=1.0,
        weight_ti=0.1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
        ti_connectivity=8,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_ti = weight_ti
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # Your anatomy:
        # 1 can touch 0
        # 2 can touch 0
        # 1 cannot touch 3
        #
        # This is the key topology prior that should be enforced.
        self.ti = TI_Loss_2D_Adjacency(
            connectivity=ti_connectivity,
            forbidden_pairs=((1, 3),),
            apply_nonlin=softmax_helper_dim1,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        net_output: (B,C,H,W)
        target:     (B,1,H,W)
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore_label only supported for target shape (B,1,H,W)"
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )

        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        ti_loss = self.ti(net_output, target) if self.weight_ti != 0 else 0

        result = (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_ti * ti_loss
        )
        return result
    




