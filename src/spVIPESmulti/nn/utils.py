import torch
import torch.nn as nn
import torch.nn.init as init


def one_hot(index: torch.Tensor, n_cat: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """One hot a tensor of categories.

    The output dtype is configurable so callers using mixed-precision training
    (autocast / bf16 / fp16) can avoid an implicit upcast on every step.
    """
    onehot = torch.zeros(index.size(0), n_cat, device=index.device, dtype=dtype)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot


def kaiming_init(m):
    """
    Initialize the parameters of a PyTorch module using Kaiming initialization.

    Parameters
    ----------
        m (nn.Module): The PyTorch module for which to initialize the parameters.

    Notes
    -----
    BatchNorm layers are intentionally skipped: ``Encoder.lvar_encoder`` relies
    on a hand-set post-BN bias of -1.0 to keep the posterior tight at init (see
    ``nn/networks.py`` and PR #31). Resetting BN bias to 0 here would silently
    re-introduce the early-training KL spike that fix prevents.

    Caller is responsible for seeding the RNG before applying this function;
    seeding inside the call would reset the global RNG on every layer.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
