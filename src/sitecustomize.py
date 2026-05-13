"""Repo-local Python startup guard.

Python imports this module automatically when the editable ``src`` directory is
on ``sys.path``. Keep this file stdlib-only: it runs before user code and should
not import ``spVIPESmulti`` or any heavy scientific dependency.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

ALLOW_USER_SITE_ENV = "SPVIPESMULTI_ALLOW_USER_SITE"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path: str) -> Path:
    try:
        expanded = Path(path).expanduser()
    except RuntimeError:
        expanded = Path(path)

    try:
        return expanded.resolve()
    except (OSError, RuntimeError):
        return expanded.absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _get_user_site_roots() -> tuple[Path, ...]:
    try:
        raw_user_sites = site.getusersitepackages()
    except (AttributeError, OSError, RuntimeError):
        raw_user_sites = ()

    if isinstance(raw_user_sites, str):
        user_sites = (raw_user_sites,)
    else:
        user_sites = tuple(raw_user_sites)

    roots: list[Path] = []
    for user_site in user_sites:
        if user_site:
            roots.append(_resolve(user_site))

    try:
        user_base = site.getuserbase()
    except (AttributeError, OSError, RuntimeError):
        user_base = ""

    if user_base:
        roots.append(_resolve(user_base))

    return tuple(dict.fromkeys(roots))


def _path_is_under_roots(path: str, roots: tuple[Path, ...]) -> bool:
    if not path:
        return False

    resolved = _resolve(path)
    return any(resolved == root or _is_relative_to(resolved, root) for root in roots)


def _apply_conda_prefix_guard() -> None:
    """Align inherited conda env vars with the running interpreter."""
    prefix = _resolve(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        return

    current = os.environ.get("CONDA_PREFIX")
    if not current or _resolve(current) != prefix:
        os.environ["CONDA_PREFIX"] = str(prefix)
        os.environ["CONDA_DEFAULT_ENV"] = prefix.name


def _apply_user_site_guard() -> None:
    if _env_flag_enabled(ALLOW_USER_SITE_ENV):
        return

    roots = _get_user_site_roots()
    if not roots:
        return

    sys.path[:] = [entry for entry in sys.path if not _path_is_under_roots(entry, roots)]
    site.ENABLE_USER_SITE = False

    # Propagate the isolation policy to child Python processes started by notebooks
    # or benchmark scripts.
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


_apply_conda_prefix_guard()
_apply_user_site_guard()
