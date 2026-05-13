"""spVIPESmulti setup file. setup file adapter from the scvi-tools-skeleton repo."""

from ._siteguard import apply_user_site_guard as _apply_user_site_guard

_apply_user_site_guard()
del _apply_user_site_guard

import logging

from rich.console import Console
from rich.logging import RichHandler

# this import needs to come after prior imports to prevent circular import
from . import data, interventions, metrics, model, module, nn, pl, traversal, utils

__all__ = ["data", "interventions", "metrics", "model", "module", "nn", "pl", "traversal", "utils"]

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "spVIPESmulti"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("spVIPESmulti: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False
