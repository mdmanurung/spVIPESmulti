from __future__ import annotations


def test_dataloaders_all_exports_are_importable() -> None:
    """Every name listed in dataloaders.__all__ should exist on the module."""

    import spVIPESmulti.dataloaders as dataloaders

    missing = [name for name in dataloaders.__all__ if not hasattr(dataloaders, name)]

    assert missing == []
