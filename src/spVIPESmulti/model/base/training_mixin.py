"""
Mixin classes for pre-coded features.
For more details on Mixin classes, see
https://docs.scvi-tools.org/en/0.9.0/user_guide/notebooks/model_user_guide.html#Mixing-in-pre-coded-features
"""


from typing import Optional

import numpy as np
from scvi.train import TrainingPlan, TrainRunner
from scvi.train._trainrunner import TrainRunner as OrigTrainRunner
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
        if version.parse(lightning_version) >= version.parse("2.0.0"):
            try:
                self.trainer.fit(
                    self.training_plan,
                    train_dataloaders=self.data_splitter,
                    ckpt_path=getattr(self, "ckpt_path", None),
                )
            except TypeError as e:
                raise RuntimeError(f"PatchedTrainRunner: Trainer.fit argument mismatch (Lightning {lightning_version}): {e}\n"
                                   f"training_plan={type(self.training_plan)}, data_splitter={type(self.data_splitter)}")
        else:
            try:
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


class MultiGroupTrainingMixin:
    """General methods for multigroup learning."""

    def train(
        self,
        group_indices_list: list[list[int]],
        batch_size: Optional[int] = 128,
        max_epochs: Optional[int] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 400,
        **trainer_kwargs,
    ) -> None:
        """
        Train a multigroup spVIPESmulti model.

        This method trains the model using a custom data splitter that handles
        multiple groups of cells separately while maintaining the shared-private
        latent space learning objective.

        Parameters
        ----------
        group_indices_list : list[list[int]]
            List of indices corresponding to each group of samples. Each inner list
            contains the indices for cells belonging to that specific group.
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
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400]).item()

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        update_dict = {
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        data_splitter = MultiGroupDataSplitter(
            self.adata_manager,
            group_indices_list=group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        runner = PatchedTrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **trainer_kwargs,
        )
        return runner()
