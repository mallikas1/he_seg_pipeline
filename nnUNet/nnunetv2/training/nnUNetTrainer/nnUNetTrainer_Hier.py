# file: nnunetv2/training/nnUNetTrainer/nnUNetTrainer_Hier.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.utilities.default_n_proc_DA import default_num_processes
from nnunetv2.training.loss.hierarchical_loss import HierarchicalDiceCE


class nnUNetTrainer_Hier(nnUNetTrainer):
    def initialize(self):
        super().initialize()

        self.loss = HierarchicalDiceCE(
            lambda_hier=0.3,
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights_ce=[0.0, 1.0, 2.0, 1.0],
            per_class_dice_weights=[0.0, 1.0, 2.5, 1.0],
        )

    def _compute_loss(self, input, target):
        total = self.loss(input, target)
        # total, terms = self.loss(input, target)
        # self.print_to_log_file({k: float(v) for k, v in terms.items()})
        return total

