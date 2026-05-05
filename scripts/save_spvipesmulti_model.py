"""Save a trained spVIPESmulti model from an active Python/IPython session.

Usage from a notebook cell:
    %run scripts/save_spvipesmulti_model.py --model-var model_spv --output-dir docs/notebooks/spvipesmulti_model --overwrite

Usage from Python:
    from scripts.save_spvipesmulti_model import save_spvipesmulti_model
    save_spvipesmulti_model(model_spv, "docs/notebooks/spvipesmulti_model", overwrite=True)
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any


def _resolve_model_from_ipython(model_var: str) -> Any:
    """Resolve a model object by variable name from an active IPython namespace."""
    try:
        from IPython import get_ipython
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "IPython is not available. Import and call save_spvipesmulti_model(model, output_dir) directly."
        ) from exc

    ip = get_ipython()
    if ip is None or not hasattr(ip, "user_ns"):
        raise RuntimeError(
            "No active IPython session detected. "
            "Use this script from a notebook with %run, or call the save function directly from Python."
        )

    if model_var not in ip.user_ns:
        raise RuntimeError(
            f"Variable '{model_var}' was not found in the notebook namespace. "
            "Train/create your model first and pass the correct --model-var."
        )
    return ip.user_ns[model_var]


def save_spvipesmulti_model(
    model: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    save_anndata: bool = True,
) -> Path:
    """Persist a trained spVIPESmulti model to disk.

    The helper adapts to different scvi-tools save signatures by only passing
    kwargs supported by `model.save`.
    """
    if not hasattr(model, "save"):
        raise TypeError("Provided object does not define a 'save' method.")

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    save_signature = inspect.signature(model.save)
    kwargs = {}
    if "overwrite" in save_signature.parameters:
        kwargs["overwrite"] = overwrite
    if "save_anndata" in save_signature.parameters:
        kwargs["save_anndata"] = save_anndata

    model.save(str(destination), **kwargs)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a trained spVIPESmulti model from notebook variables.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the model files will be written.",
    )
    parser.add_argument(
        "--model-var",
        default="model_spv",
        help="Notebook variable name containing the trained model object (default: model_spv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target directory if it already exists.",
    )
    parser.add_argument(
        "--no-save-anndata",
        action="store_true",
        help="Do not save AnnData alongside model weights when supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = _resolve_model_from_ipython(args.model_var)
    output_path = save_spvipesmulti_model(
        model,
        args.output_dir,
        overwrite=args.overwrite,
        save_anndata=not args.no_save_anndata,
    )
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
