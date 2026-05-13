"""Runtime isolation helpers for local development environments."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from types import ModuleType

ALLOW_USER_SITE_ENV = "SPVIPESMULTI_ALLOW_USER_SITE"
RISKY_USER_SITE_MODULES = {
    "anndata",
    "lightning",
    "mudata",
    "pytorch_lightning",
    "scanpy",
    "scvi",
    "torch",
    "torchmetrics",
    "torchvision",
}


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
        raw_user_sites: str | tuple[str, ...] = site.getusersitepackages()
    except (AttributeError, OSError, RuntimeError):
        raw_user_sites = ()

    user_sites: tuple[str, ...]
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


def _module_file(module: ModuleType) -> str:
    module_file = getattr(module, "__file__", None)
    return module_file if isinstance(module_file, str) else ""


def _loaded_modules_from_user_site(roots: tuple[Path, ...]) -> list[str]:
    loaded: list[str] = []
    for module_name, module in sys.modules.items():
        top_level_name = module_name.partition(".")[0]
        if top_level_name not in RISKY_USER_SITE_MODULES or module is None:
            continue

        module_file = _module_file(module)
        if module_file and _path_is_under_roots(module_file, roots):
            loaded.append(f"{module_name} ({module_file})")

    return loaded


def apply_user_site_guard(*, fail_on_loaded: bool = True) -> tuple[str, ...]:
    """Remove Python user-site paths before importing scientific dependencies.

    HPC notebook kernels can accidentally mix conda packages with packages from
    ``~/.local``. For torch/scvi/lightning stacks this often manifests as opaque
    binary-extension errors such as ``operator torchvision::nms does not exist``.
    """
    _apply_conda_prefix_guard()

    if _env_flag_enabled(ALLOW_USER_SITE_ENV):
        return ()

    roots = _get_user_site_roots()
    if not roots:
        return ()

    original_path = list(sys.path)
    filtered_path = [entry for entry in original_path if not _path_is_under_roots(entry, roots)]
    removed = tuple(entry for entry in original_path if entry not in filtered_path)

    if removed:
        sys.path[:] = filtered_path

    site.ENABLE_USER_SITE = False
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    if fail_on_loaded:
        loaded = _loaded_modules_from_user_site(roots)
        if loaded:
            module_list = "\n".join(f"- {module}" for module in sorted(loaded))
            raise RuntimeError(
                "spVIPESmulti detected core scientific packages already imported from "
                "the Python user-site directory. Restart the Python process with "
                "PYTHONNOUSERSITE=1 or select the Python (spvm) notebook kernel.\n"
                f"{module_list}"
            )

    return removed
