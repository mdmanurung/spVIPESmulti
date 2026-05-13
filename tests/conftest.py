"""Test-session compatibility shims."""

from __future__ import annotations


def _ensure_torchvision_nms_schema() -> None:
    """Avoid import-time failures when torchvision lacks the optional nms op."""
    try:
        import torchvision.extension  # noqa: F401

        return
    except (ImportError, OSError, RuntimeError):
        pass

    try:
        import torch

        if torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta"):
            return
    except RuntimeError:
        pass
    except (ImportError, OSError):
        return

    try:
        from torch.library import Library

        lib = Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        globals()["_TORCHVISION_COMPAT_LIB"] = lib
    except (ImportError, OSError, RuntimeError):
        return


_ensure_torchvision_nms_schema()
