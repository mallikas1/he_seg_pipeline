import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalConstrainedLoss(nn.Module):
    def __init__(self, boundary_weight=2.0):
        super(HierarchicalConstrainedLoss, self).__init__()
        self.boundary_weight = boundary_weight

    def forward(self, predictions, targets):
        # Dice or CrossEntropy loss for primary segmentation accuracy
        main_loss = F.cross_entropy(predictions, targets)

        # Boundary Constraints for Fascicle-Perineurium and Perineurium-Epineurium
        fascicle_loss = self.boundary_constraint(predictions, targets, 1, 2)  # Fascicle-Perineurium
        perineurium_loss = self.boundary_constraint(predictions, targets, 2, 3)  # Perineurium-Epineurium

        # No Overlap Constraint: Penalize any epineurium enclosed by perineurium
        exclusion_loss = self.no_overlap_constraint(predictions, targets, 2, 3)

        # Combine all loss terms
        total_loss = main_loss + self.boundary_weight * (fascicle_loss + perineurium_loss + exclusion_loss)
        return total_loss

    def boundary_constraint(self, predictions, targets, inner_label, outer_label):
        # Identify regions where inner_label should be enclosed by outer_label
        inner_boundary = (targets == inner_label).float()
        outer_boundary = (targets == outer_label).float()

        # Penalize cases where inner regions are not fully enclosed by outer regions
        boundary_loss = F.binary_cross_entropy_with_logits(predictions, outer_boundary, weight=inner_boundary)
        return boundary_loss

    def no_overlap_constraint(self, predictions, targets, inner_label, exclusion_label):
        # Penalize any overlap of exclusion_label within the inner_label region
        inner_region = (targets == inner_label).float()
        exclusion_region = (predictions == exclusion_label).float()

        # Create an overlap loss for cases where exclusion is found within inner region
        overlap_loss = (inner_region * exclusion_region).sum() / (inner_region.sum() + 1e-6)
        return overlap_loss






import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalLoss2(nn.Module):
    """
    Adds soft hierarchical constraints on top of a base segmentation loss.
    Works with logits of shape [B, C, H, W] (or [B, C, D, H, W]).
    class_idx: dict like {"bg":0, "fascicle":1, "perineurium":2, "epineurium":3, "vessel":4}
    containment_pairs: list of (inner, outer) class ids to enforce p_inner <= p_outer
    forbid_inside: list of (inner_region, forbidden_class) to penalize forbidden inside inner_region
    """
    def __init__(
        self,
        class_idx: dict,
        containment_pairs=((1, 2), (2, 3)),   # fascicle⊂perineurium, perineurium⊂epineurium
        forbid_inside=((2, 3),),              # no epineurium inside perineurium
        lambda_contain=0.3,
        lambda_forbid=0.2,
        use_target_focus=True,                # focus constraints where GT inner exists
        ignore_index: int | None = None
    ):
        super().__init__()
        self.ci = class_idx
        self.containment_pairs = containment_pairs
        self.forbid_inside = forbid_inside
        self.lc = float(lambda_contain)
        self.lf = float(lambda_forbid)
        self.use_target_focus = use_target_focus
        self.ignore_index = ignore_index

    def forward(self, logits, targets, base_loss=None):
        """
        logits: [B, C, ...]; targets: [B, ...] (Long)
        base_loss: a callable (logits, targets) -> scalar; if None, uses CrossEntropy
        """
        if base_loss is None:
            main = F.cross_entropy(
                logits, targets,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100
            )
        else:
            main = base_loss(logits, targets)

        probs = F.softmax(logits, dim=1)  # [B, C, ...]
        dims = list(range(2, probs.ndim)) # spatial dims

        # Optional focus mask to avoid enforcing constraints on pure background
        if self.use_target_focus:
            # mask where target equals given inner class
            def focus_mask(cls):
                m = (targets == cls).float()
                if self.ignore_index is not None:
                    m = m * (targets != self.ignore_index).float()
                return m
        else:
            def focus_mask(cls):
                return torch.ones_like(targets, dtype=torch.float32, device=targets.device)

        contain_loss = 0.0
        for inner, outer in self.containment_pairs:
            p_in  = probs[:, inner]
            p_out = probs[:, outer]
            m = focus_mask(inner)
            # Penalize places where inner prob exceeds outer prob
            # L = mean( relu(p_in - p_out) ) optionally masked
            diff = torch.relu(p_in - p_out)
            if self.use_target_focus:
                num = (diff * m).sum()
                den = m.sum().clamp_min(1.0)
                contain_loss = contain_loss + num / den
            else:
                contain_loss = contain_loss + diff.mean()

        forbid_loss = 0.0
        for inner_region, forbidden_class in self.forbid_inside:
            p_forbid = probs[:, forbidden_class]
            m = focus_mask(inner_region)
            # Penalize forbidden probability inside inner region
            if self.use_target_focus:
                num = (p_forbid * m).sum()
                den = m.sum().clamp_min(1.0)
                forbid_loss = forbid_loss + num / den
            else:
                forbid_loss = forbid_loss + p_forbid.mean()

        total = main + self.lc * contain_loss + self.lf * forbid_loss
        return total
