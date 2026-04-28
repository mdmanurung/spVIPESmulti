"""Utilities for splitting a dataset into training, validation, and test set."""

from math import ceil, floor
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from scvi import settings

from spVIPES.data._manager import AnnDataManager
from spVIPES.dataloaders._concat_dataloader import ConcatDataLoader


def _validate_data_split(n_samples: int, train_size: float, validation_size: Optional[float] = None) -> tuple[int, int]:
    """Compute (n_train, n_val) for a split, vendored from scvi-tools to avoid a private import."""
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")
    n_train = ceil(train_size * n_samples)
    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0:
        raise ValueError("Invalid validation_size. Must be: 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1.0:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)
    if n_train == 0:
        raise ValueError(f"With n_samples={n_samples}, train_size={train_size}, the resulting train set is empty.")
    return n_train, n_val


# accessed https://github.com/Genentech/multiGroupVI/blob/main/multigroup_vi/data/data_splitting.py 01/04/2023
class MultiGroupDataSplitter(pl.LightningDataModule):
    """
    Create MultiGroupDataLoader for training, validation, and test set.
    Args:
    ----
        adata_manager: AnnDataManager object registered via `setup_anndata`.
        group_indices_list: [[species_1 obs indices], [species_2 obs indices]]
        train_size: Proportion of data to include in the training set.
        validation_size: Proportion of data to include in the validation set. The
            remaining proportion after `train_size` and `validation_size` is used for
            the test set.
        **kwargs: Keyword args for data loader (`MultiGroupDataLoader`).
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_indices_list: list[list[int]],
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.adata_manager = adata_manager
        self.group_indices_list = group_indices_list
        self.train_size = train_size
        self.validation_size = validation_size
        self.data_loader_kwargs = kwargs

        self.n_per_group = [len(group_indices) for group_indices in group_indices_list]

        n_train_per_group = []
        n_val_per_group = []

        for group_indices in group_indices_list:
            n_train, n_val = _validate_data_split(len(group_indices), self.train_size, self.validation_size)
            n_train_per_group.append(n_train)
            n_val_per_group.append(n_val)

        self.n_val_per_group = n_val_per_group
        self.n_train_per_group = n_train_per_group

    def setup(self, stage: Optional[str] = None):
        random_state = np.random.RandomState(seed=settings.seed)

        self.train_idx_per_group = []
        self.val_idx_per_group = []
        self.test_idx_per_group = []

        for i, group_indices in enumerate(self.group_indices_list):
            group_permutation = random_state.permutation(group_indices)
            n_train_group = self.n_train_per_group[i]
            n_val_group = self.n_val_per_group[i]

            self.val_idx_per_group.append(group_permutation[:n_val_group])
            self.train_idx_per_group.append(group_permutation[n_val_group : (n_val_group + n_train_group)])
            self.test_idx_per_group.append(group_permutation[(n_val_group + n_train_group) :])

        self.pin_memory = settings.dl_pin_memory_gpu_training and torch.cuda.is_available()
        self.train_idx = self.train_idx_per_group
        self.test_idx = self.test_idx_per_group
        self.val_idx = self.val_idx_per_group

    def _get_multigroup_dataloader(
        self,
        group_indices_list,
    ) -> ConcatDataLoader:
        return ConcatDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def train_dataloader(self) -> ConcatDataLoader:
        return self._get_multigroup_dataloader(self.train_idx_per_group)

    def val_dataloader(self) -> ConcatDataLoader:
        if np.all([len(val_idx) > 0 for val_idx in self.val_idx_per_group]):
            return self._get_multigroup_dataloader(self.val_idx_per_group)
        else:
            pass

    def test_dataloader(self) -> ConcatDataLoader:
        if np.all([len(test_idx) > 0 for test_idx in self.test_idx_per_group]):
            return self._get_multigroup_dataloader(self.test_idx_per_group)
        else:
            pass
