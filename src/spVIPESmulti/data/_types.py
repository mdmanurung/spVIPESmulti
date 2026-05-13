"""Type aliases vendored from scvi-tools to avoid private-API imports."""

from typing import Literal

from anndata import AnnData
from mudata import MuData

AnnOrMuData = AnnData | MuData
MinifiedDataType = Literal["latent_posterior_parameters"]
