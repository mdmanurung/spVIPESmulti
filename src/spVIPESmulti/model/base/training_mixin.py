"""
Mixin classes for pre-coded features.
For more details on Mixin classes, see
https://docs.scvi-tools.org/en/0.9.0/user_guide/notebooks/model_user_guide.html#Mixing-in-pre-coded-features
"""


from typing import Literal, Optional
import warnings

import numpy as np
from scvi.train import TrainingPlan, TrainRunner
from scvi.train._trainrunner import TrainRunner as OrigTrainRunner


class SpVIPESmultiTrainingPlan(TrainingPlan):
    """TrainingPlan subclass with extended LR scheduler support.

    Adds two capabilities on top of scvi's TrainingPlan:

    1. ``lr_scheduler_type="cosine"``: replaces ReduceLROnPlateau with
       CosineAnnealingLR so the LR decays on a fixed schedule regardless of
       whether the monitored metric plateaus.  Use when the loss is still
       declining throughout training and plateau detection never fires.

    2. Validation-frequency alignment for ReduceLROnPlateau: when
       ``check_val_every_n_epoch > 1``, Lightning only logs validation metrics
       every N epochs.  Adding ``frequency=N`` to the scheduler config prevents
       a MisconfigurationException at epoch 1 when the monitored metric is not
       yet available.
    """

    def __init__(
        self,
        *args,
        check_val_every_n_epoch: int = 1,
        lr_scheduler_type: Literal["plateau", "cosine"] = "plateau",
        lr_cosine_T_max: int = 400,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._check_val_every_n_epoch = check_val_every_n_epoch
        self._lr_scheduler_type = lr_scheduler_type
        self._lr_cosine_T_max = lr_cosine_T_max

    def configure_optimizers(self):
        config = super().configure_optimizers()

        if self._lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                config["optimizer"],
                T_max=self._lr_cosine_T_max,
                eta_min=self.lr_min,
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        else:
            # plateau: align stepping frequency with validation cadence
            freq = self._check_val_every_n_epoch
            if freq > 1:
                lr_cfg = config.get("lr_scheduler")
                if isinstance(lr_cfg, dict):
                    lr_cfg["frequency"] = freq

        return config


class PatchedTrainRunner(OrigTrainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        import lightning as pl
        from packaging import version
        # Validate training_plan
        if not hasattr(self, "training_plan") or self.training_plan is None:
            raise RuntimeError("PatchedTrainRunner: training_plan is not set. Ensure TrainingPlan is constructed correctly.")
        # Validate data_splitter
        if not hasattr(self, "data_splitter") or self.data_splitter is None:
            raise RuntimeError("PatchedTrainRunner: data_splitter is not set. Ensure DataSplitter is constructed correctly.")

        # Pre-fit: propagate dataset sizes to the training plan
        if hasattr(self.data_splitter, "n_train"):
            self.training_plan.n_obs_training = self.data_splitter.n_train
        if hasattr(self.data_splitter, "n_val"):
            self.training_plan.n_obs_validation = self.data_splitter.n_val

        # Lightning-version-aware trainer.fit() call
        lightning_version = pl.__version__
        n_val = getattr(self.data_splitter, "n_val", 0)
        
        # Warn if validation checking was requested but there's no validation data
        if n_val == 0 and getattr(self.trainer, "check_val_every_n_epoch", 1) > 1:
            warnings.warn(
                f"check_val_every_n_epoch={self.trainer.check_val_every_n_epoch} was set, but "
                "train_size=1.0 leaves no validation data. Validation checking will be skipped.",
                UserWarning,
                stacklevel=3,
            )
        
        if version.parse(lightning_version) >= version.parse("2.0.0"):
            try:
                if n_val == 0:
                    # Lightning 2.6.x rejects a datamodule whose val_dataloader()
                    # returns None, so use the explicit train-dataloader path when
                    # the requested split leaves no validation cells.
                    self.data_splitter.setup("fit")
                    self.trainer.fit(
                        self.training_plan,
                        train_dataloaders=self.data_splitter.train_dataloader(),
                        val_dataloaders=None,  # Explicitly set to None to prevent Lightning from calling val_dataloader()
                        ckpt_path=getattr(self, "ckpt_path", None),
                    )
                else:
                    self.trainer.fit(
                        self.training_plan,
                        datamodule=self.data_splitter,
                        ckpt_path=getattr(self, "ckpt_path", None),
                    )
            except TypeError as e:
                raise RuntimeError(f"PatchedTrainRunner: Trainer.fit argument mismatch (Lightning {lightning_version}): {e}\n"
                                   f"training_plan={type(self.training_plan)}, data_splitter={type(self.data_splitter)}")
        else:
            try:
                if n_val == 0:
                    self.data_splitter.setup("fit")
                    self.trainer.fit(
                        self.training_plan,
                        self.data_splitter.train_dataloader(),
                        ckpt_path=getattr(self, "ckpt_path", None),
                    )
                else:
                    self.trainer.fit(
                        self.training_plan,
                        self.data_splitter,
                        ckpt_path=getattr(self, "ckpt_path", None),
                    )
            except TypeError as e:
                raise RuntimeError(f"PatchedTrainRunner: Trainer.fit argument mismatch (Lightning {lightning_version}): {e}\n"
                                   f"training_plan={type(self.training_plan)}, data_splitter={type(self.data_splitter)}")

        # Post-fit bookkeeping (mirrors TrainRunner.__call__)
        self._update_history()
        self.model.train_indices = getattr(self.data_splitter, "train_idx", None)
        self.model.test_indices = getattr(self.data_splitter, "test_idx", None)
        self.model.validation_indices = getattr(self.data_splitter, "val_idx", None)
        self.model.module.eval()
        self.model.is_trained_ = True
        self.model.to_device(self.device)
        self.model.trainer = self.trainer

from spVIPESmulti.data._multi_datasplitter import MultiGroupDataSplitter
from spVIPESmulti.utils import resolve_group_indices_list


class MultiGroupTrainingMixin:
    """General methods for multigroup learning."""

    def train(
        self,
        group_indices_list: Optional[list[list[int]]] = None,
        batch_size: Optional[int] = 128,
        max_epochs: Optional[int] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 400,
        num_workers: int = 0,
        **trainer_kwargs,
    ) -> None:
        """
        Train a multigroup spVIPESmulti model.

        This method trains the model using a custom data splitter that handles
        multiple groups of cells separately while maintaining the shared-private
        latent space learning objective.

        Parameters
        ----------
        group_indices_list : list[list[int]], optional
            List of indices corresponding to each group of samples. Each inner list
            contains the indices for cells belonging to that specific group. If
            ``None``, values are inferred from ``adata.uns['groups_obs_indices']``.
        max_epochs : int, optional
            Number of passes through the dataset. If None, defaults to
            ``np.min([round((20000 / n_cells) * 400), 400])``.
        train_size : float, default=0.9
            Size of training set in the range [0.0, 1.0].
        validation_size : float, optional
            Size of the validation set. If None, defaults to ``1 - train_size``.
            If ``train_size + validation_size < 1``, the remaining cells belong
            to the test set.
        batch_size : int, default=128
            Mini-batch size to use during training.
        early_stopping : bool, default=False
            Whether to perform early stopping. Additional arguments can be passed
            in ``**trainer_kwargs``.
        plan_kwargs : dict, optional
            Keyword arguments for the training plan. Arguments passed to ``train()``
            will overwrite values present in ``plan_kwargs``, when appropriate.
        n_steps_kl_warmup : int, optional
            Number of training steps for KL warmup. Takes precedence over n_epochs_kl_warmup.
        n_epochs_kl_warmup : int, default=400
            Number of epochs for KL divergence warmup.
        **trainer_kwargs
            Additional keyword arguments forwarded to ``pl.Trainer`` via scvi-tools'
            ``TrainRunner``. To select an accelerator (replaces the removed ``use_gpu``
            argument), pass e.g. ``accelerator="gpu"`` and ``devices=1``.
        num_workers : int, default=0
            Number of worker processes for the DataLoader. ``0`` means data is
            loaded in the main process (safest, compatible with all platforms).
            Set to a positive integer (e.g. ``4``) on multi-core HPC nodes with
            a GPU to overlap data loading with forward/backward passes.
            Requires that your dataset supports multi-process access (standard
            AnnData on disk does).

        Returns
        -------
        None
            The model is trained in-place.

        Notes
        -----
        This method uses a specialized MultiGroupDataSplitter that ensures proper
        handling of multiple cell groups during training, maintaining the integrity
        of the shared-private latent space learning.
        """
        group_indices_list, inferred = resolve_group_indices_list(self.adata, group_indices_list)
        if inferred:
            self._warn_group_indices_auto_inferred("train")

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400]).item()

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        update_dict = {
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        plan_kwargs.update(update_dict)

        # When using cosine annealing, default T_max to the training horizon so
        # the LR reaches eta_min exactly at the last epoch.
        if plan_kwargs.get("lr_scheduler_type") == "cosine":
            plan_kwargs.setdefault("lr_cosine_T_max", max_epochs)

        data_splitter = MultiGroupDataSplitter(
            self.adata_manager,
            group_indices_list=group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        cvene = trainer_kwargs.get("check_val_every_n_epoch", 1)
        training_plan = SpVIPESmultiTrainingPlan(self.module, check_val_every_n_epoch=cvene, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        if data_splitter.n_val > 0 and "check_val_every_n_epoch" not in trainer_kwargs:
            trainer_kwargs["check_val_every_n_epoch"] = 1
        # Default to clipping by global L2 norm to bound early-epoch gradient
        # spikes (e.g. KL blowup); user can override via trainer_kwargs.
        trainer_kwargs.setdefault("gradient_clip_val", 1.0)
        trainer_kwargs.setdefault("gradient_clip_algorithm", "norm")
        runner = PatchedTrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **trainer_kwargs,
        )
        return runner()

    def _warn_group_indices_auto_inferred(self, caller: str) -> None:
        """Emit a one-time informational warning when group indices are inferred."""
        if getattr(self, "_group_indices_auto_infer_warned", False):
            return
        warnings.warn(
            "group_indices_list was not provided to "
            f"{caller}(); inferred from adata.uns['groups_obs_indices'].",
            UserWarning,
            stacklevel=2,
        )
        self._group_indices_auto_infer_warned = True
