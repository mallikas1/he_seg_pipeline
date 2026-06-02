import os
import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_ce_boundary import DC_CE_Boundary_loss
from nnunetv2.training.loss.dice import SoftDiceLoss  # IMPORTANT
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # optional

# make sure these exist in your codebase/import path
# from nnunetv2.training.loss.dice_ce_boundary import DC_CE_Boundary_loss2
# from nnunetv2.utilities.helpers import softmax_helper_dim1

# project = os.getenv("WANDB_PROJECT", "vagus-nnunet-baseline")
# entity = os.getenv("WANDB_ENTITY", "mallikafnu-cwru")
def _safe_get(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return default



class nnUNetTrainerWandB(nnUNetTrainer):
    """
    Version-robust W&B logging trainer for nnU-Net v2.
    Avoids assuming dataset_name exists.
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._wandb_run = None
        self._wandb_enabled = True

    def initialize(self):
        super().initialize()

        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            self._wandb_enabled = False
            return

        project = os.getenv("WANDB_PROJECT", "nnunetv2")
        entity = os.getenv("WANDB_ENTITY", None)

        dataset_id = _safe_get(self, ["dataset_name_or_id", "dataset_id"], default=None)
        dataset_json_path = str(_safe_get(self, ["dataset_json"], default=""))
        output_folder = str(_safe_get(self, ["output_folder"], default=""))

        dataset_tag = dataset_id or os.path.basename(os.path.dirname(dataset_json_path)) or "dataset"

        cfg_name = _safe_get(self, ["configuration", "configuration_name"], default="unknown_cfg")
        fold = _safe_get(self, ["fold"], default="NA")

        # run_name = os.getenv("WANDB_RUN_NAME", f"{dataset_tag}-{cfg_name}-fold{fold}")
        # group = os.getenv("WANDB_GROUP", f"{dataset_tag}_{cfg_name}")

        run_name = os.getenv("WANDB_RUN_NAME") or os.getenv("WANDB_NAME") \
           or f"{dataset_tag}-{cfg_name}-fold{fold}"

        group = os.getenv("WANDB_GROUP") or os.getenv("WANDB_RUN_GROUP") \
            or f"{dataset_tag}_{cfg_name}"

        cfg = {
            "dataset_tag": dataset_tag,
            "configuration": str(cfg_name),
            "fold": int(fold) if str(fold).isdigit() else str(fold),
            "trainer": type(self).__name__,
            "dataset_json": dataset_json_path,
            "output_folder": output_folder,
        }

        # patch size/batch size best-effort
        patch_size = _safe_get(self, ["patch_size"], None)
        if patch_size is None:
            cm = _safe_get(self, ["configuration_manager"], None)
            patch_size = _safe_get(cm, ["patch_size"], None) if cm is not None else None
        if patch_size is not None:
            cfg["patch_size"] = list(patch_size)

        cm = _safe_get(self, ["configuration_manager"], None)
        bs = _safe_get(cm, ["batch_size"], None) if cm is not None else None
        if bs is not None:
            cfg["batch_size"] = int(bs)

        self._wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config=cfg,
            reinit=True,
        )


    def on_epoch_end(self):
        super().on_epoch_end()
        if not self._wandb_enabled or self._wandb_run is None:
            return

        # Epoch number
        epoch = _safe_get(self, ["current_epoch"], default=None)
        metrics = {"epoch": epoch}

        # LR (best-effort)
        try:
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                metrics["lr"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            pass

        # nnU-Net logger (varies by version)
        try:
            log = getattr(self.logger, "my_fantastic_logging", None)
            if isinstance(log, dict):
                for k, v in log.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        metrics[str(k)] = v[-1]
        except Exception:
            pass

        # Don’t ever crash training because of wandb
        try:
            wandb.log(metrics, step=epoch)
        except Exception:
            pass

    def on_training_end(self):
        super().on_training_end()
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass





class nnUNetTrainerWandB_loss(nnUNetTrainer):
    """
    nnU-Net v2 trainer:
    - defines loss via _build_loss() (so deep supervision works)
    - logs to W&B on epoch end
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._wandb_run = None
        self._wandb_enabled = True

    # -------------------------
    # Loss (nnU-Net v2 pattern)
    # -------------------------
    def _build_loss(self):
        # Use nnU-Net config knobs so your loss matches the run configuration
        soft_dice_kwargs = {
            "batch_dice": getattr(self.configuration_manager, "batch_dice", True),
            "smooth": 1e-5,
            "do_bg": True,                 # change to False if you don't want background dice
            "ddp": getattr(self, "is_ddp", False),
        }
        ce_kwargs = {}

        loss = DC_CE_Boundary_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            weight_boundary=0.5,
            boundary_boost=4.0,
            boundary_kernel=3,
            ignore_label=getattr(self.label_manager, "ignore_label", None),
            dice_class=MemoryEfficientSoftDiceLoss,  # important for now
            peri_only=False,
            # peri_only=True,
            peri_label_id=2,  # <-- make sure this matches your dataset
        )

        # Deep supervision wrapping the nnU-Net way
        if getattr(self, "enable_deep_supervision", False):
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)

            # DDP + no torch.compile: avoid "unused params" crash when last weight is 0
            if getattr(self, "is_ddp", False) and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0.0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        print("Using custom DC+CE Boundary loss | deep supervision:", getattr(self, "enable_deep_supervision", False))
        return loss

    # -------------------------
    # W&B init (after nnU-Net init)
    # -------------------------
    def initialize(self):
        # This will call self._build_loss() internally and set self.loss correctly
        super().initialize()

        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            self._wandb_enabled = False
            return

        project = os.getenv("WANDB_PROJECT", "nnunetv2")
        entity = os.getenv("WANDB_ENTITY", None)

        dataset_id = _safe_get(self, ["dataset_name_or_id", "dataset_id"], default=None)
        dataset_json_path = str(_safe_get(self, ["dataset_json"], default=""))
        output_folder = str(_safe_get(self, ["output_folder"], default=""))

        dataset_tag = dataset_id or os.path.basename(os.path.dirname(dataset_json_path)) or "dataset"
        cfg_name = _safe_get(self, ["configuration", "configuration_name"], default="unknown_cfg")
        fold = _safe_get(self, ["fold"], default="NA")

        run_name = os.getenv("WANDB_RUN_NAME") or os.getenv("WANDB_NAME") \
            or f"{dataset_tag}-{cfg_name}-fold{fold}"
        group = os.getenv("WANDB_GROUP") or os.getenv("WANDB_RUN_GROUP") \
            or f"{dataset_tag}_{cfg_name}"

        cfg = {
            "dataset_tag": dataset_tag,
            "configuration": str(cfg_name),
            "fold": int(fold) if str(fold).isdigit() else str(fold),
            "trainer": type(self).__name__,
            "dataset_json": dataset_json_path,
            "output_folder": output_folder,
            "deep_supervision": bool(getattr(self, "enable_deep_supervision", False)),
            "torch_compile": bool(self._do_i_compile()) if hasattr(self, "_do_i_compile") else None,
        }

        # patch size/batch size best-effort
        patch_size = _safe_get(self, ["patch_size"], None)
        if patch_size is None:
            cm = _safe_get(self, ["configuration_manager"], None)
            patch_size = _safe_get(cm, ["patch_size"], None) if cm is not None else None
        if patch_size is not None:
            cfg["patch_size"] = list(patch_size)

        cm = _safe_get(self, ["configuration_manager"], None)
        bs = _safe_get(cm, ["batch_size"], None) if cm is not None else None
        if bs is not None:
            cfg["batch_size"] = int(bs)

        self._wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config=cfg,
            reinit=True,
        )

    def on_epoch_end(self):
        super().on_epoch_end()
        if not self._wandb_enabled or self._wandb_run is None:
            return

        epoch = _safe_get(self, ["current_epoch"], default=None)
        metrics = {"epoch": epoch}

        try:
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                metrics["lr"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            pass

        # nnU-Net logger (version-dependent)
        try:
            log = getattr(self.logger, "my_fantastic_logging", None)
            if isinstance(log, dict):
                for k, v in log.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        metrics[str(k)] = v[-1]
        except Exception:
            pass

        try:
            wandb.log(metrics, step=epoch)
        except Exception:
            pass

    def on_training_end(self):
        super().on_training_end()
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass






import os
import numpy as np
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dc_ce_topo import DC_CE_TI_loss_2D


class nnUNetTrainerWandB_Topoloss(nnUNetTrainer):
    """
    nnU-Net v2 trainer:
    - defines loss via _build_loss() (so deep supervision works)
    - logs to W&B on epoch end
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._wandb_run = None
        self._wandb_enabled = True

    # -------------------------
    # Loss (nnU-Net v2 pattern)
    # -------------------------
    def _build_loss(self):
        # This custom loss assumes standard label-map training, not region-based training
        if getattr(self.label_manager, "has_regions", False):
            raise NotImplementedError(
                "DC_CE_Topo_loss currently supports standard class-label training, "
                "not region-based training."
            )

        # Keep nnU-Net config knobs
        soft_dice_kwargs = {
            "batch_dice": False,
            "do_bg": True,
            "smooth": 1e-5,
            "ddp": False,
        }

        ce_kwargs = {}

        loss = DC_CE_TI_loss_2D(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            weight_ti=0.05,   # start small, then tune
            ignore_label=None,
            dice_class=SoftDiceLoss,
            ti_connectivity=8,
        )

        # Deep supervision wrapping the nnU-Net way
        if getattr(self, "enable_deep_supervision", False):
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))],
                dtype=np.float32
            )

            # nnU-Net DDP quirk: last weight should be tiny instead of 0 in some cases
            if getattr(self, "is_ddp", False) and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0.0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        print(
            "Using custom DC+CE+Topo loss | deep supervision:",
            getattr(self, "enable_deep_supervision", False)
        )
        return loss

    # -------------------------
    # W&B init (after nnU-Net init)
    # -------------------------
    def initialize(self):
        # This calls self._build_loss() internally and sets self.loss
        super().initialize()

        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            self._wandb_enabled = False
            return

        project = os.getenv("WANDB_PROJECT", "nnunetv2")
        entity = os.getenv("WANDB_ENTITY", None)

        dataset_id = _safe_get(self, ["dataset_name_or_id", "dataset_id"], default=None)
        dataset_json_path = str(_safe_get(self, ["dataset_json"], default=""))
        output_folder = str(_safe_get(self, ["output_folder"], default=""))

        dataset_tag = dataset_id or os.path.basename(os.path.dirname(dataset_json_path)) or "dataset"
        cfg_name = _safe_get(self, ["configuration", "configuration_name"], default="unknown_cfg")
        fold = _safe_get(self, ["fold"], default="NA")

        run_name = (
            os.getenv("WANDB_RUN_NAME")
            or os.getenv("WANDB_NAME")
            or f"{dataset_tag}-{cfg_name}-fold{fold}"
        )
        group = (
            os.getenv("WANDB_GROUP")
            or os.getenv("WANDB_RUN_GROUP")
            or f"{dataset_tag}_{cfg_name}"
        )

        cfg = {
            "dataset_tag": dataset_tag,
            "configuration": str(cfg_name),
            "fold": int(fold) if str(fold).isdigit() else str(fold),
            "trainer": type(self).__name__,
            "dataset_json": dataset_json_path,
            "output_folder": output_folder,
            "deep_supervision": bool(getattr(self, "enable_deep_supervision", False)),
            "torch_compile": bool(self._do_i_compile()) if hasattr(self, "_do_i_compile") else None,
            "loss_name": "DC_CE_Topo_loss",
            "weight_ce": 1.0,
            "weight_dice": 1.0,
            "weight_topo": 1e-3,
            "topo_connectivity": 8,
            "topo_min_thick": 1,
        }

        patch_size = _safe_get(self, ["patch_size"], None)
        if patch_size is None:
            cm = _safe_get(self, ["configuration_manager"], None)
            patch_size = _safe_get(cm, ["patch_size"], None) if cm is not None else None
        if patch_size is not None:
            cfg["patch_size"] = list(patch_size)

        cm = _safe_get(self, ["configuration_manager"], None)
        bs = _safe_get(cm, ["batch_size"], None) if cm is not None else None
        if bs is not None:
            cfg["batch_size"] = int(bs)

        self._wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config=cfg,
            reinit=True,
        )

    def on_epoch_end(self):
        super().on_epoch_end()
        if not self._wandb_enabled or self._wandb_run is None:
            return

        epoch = _safe_get(self, ["current_epoch"], default=None)
        metrics = {"epoch": epoch}

        try:
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                metrics["lr"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            pass

        try:
            log = getattr(self.logger, "my_fantastic_logging", None)
            if isinstance(log, dict):
                for k, v in log.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        metrics[str(k)] = v[-1]
        except Exception:
            pass

        try:
            wandb.log(metrics, step=epoch)
        except Exception:
            pass

    def on_training_end(self):
        super().on_training_end()
        if self._wandb_enabled and self._wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


