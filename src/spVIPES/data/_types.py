"""Type aliases vendored from scvi-tools to avoid private-API imports."""

from typing import Literal, Union

from anndata import AnnData
from mudata import MuData

AnnOrMuData = Union[AnnData, MuData]
MinifiedDataType = Literal["latent_posterior_parameters"]
