"""AST-based tests verifying scvi-tools 1.x compatibility of spVIPES source files.

These tests parse Python source files directly with the `ast` module so they
run without scvi installed.  Integration tests (marked ``integration``) require
a live scvi 1.x environment.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src" / "spVIPES"


def _collect_imports(path: pathlib.Path):
    """Return list of (node_type, module, names) for every import in *path*."""
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(("import", alias.name, []))
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            imports.append(("from", node.module or "", names))
    return imports


# ---------------------------------------------------------------------------
# Step 1 — _base_field.py: private scvi.data._constants / _utils redirected
# ---------------------------------------------------------------------------

class TestBaseFieldImports:
    PATH = SRC / "data" / "fields" / "_base_field.py"

    def test_no_scvi_data_constants_import(self):
        """from scvi.data import _constants must be gone."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from" and module == "scvi.data":
                assert "_constants" not in names, (
                    "Found 'from scvi.data import _constants' — "
                    "must redirect to 'from spVIPES.data import _constants'"
                )
            if kind == "from":
                assert not module.startswith("scvi.data._constants"), (
                    "Found import from 'scvi.data._constants' — "
                    "must use local spVIPES.data._constants"
                )

    def test_no_scvi_data_utils_get_anndata_attribute(self):
        """from scvi.data._utils import get_anndata_attribute must be gone."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from" and module.startswith("scvi.data._utils"):
                assert "get_anndata_attribute" not in names, (
                    "Found 'from scvi.data._utils import get_anndata_attribute' — "
                    "must redirect to local spVIPES.data._utils"
                )

    def test_local_constants_imported(self):
        """from spVIPES.data import _constants must be present."""
        found = any(
            kind == "from" and module == "spVIPES.data" and "_constants" in names
            for kind, module, names in _collect_imports(self.PATH)
        )
        assert found, "Missing 'from spVIPES.data import _constants' in _base_field.py"

    def test_local_utils_imported(self):
        """from spVIPES.data._utils import get_anndata_attribute must be present."""
        found = any(
            kind == "from"
            and module == "spVIPES.data._utils"
            and "get_anndata_attribute" in names
            for kind, module, names in _collect_imports(self.PATH)
        )
        assert found, "Missing 'from spVIPES.data._utils import get_anndata_attribute' in _base_field.py"


# ---------------------------------------------------------------------------
# Step 2 — _manager.py: AnnTorchDataset from public scvi.data
# ---------------------------------------------------------------------------

class TestAnnTorchDatasetImport:
    PATH = SRC / "data" / "_manager.py"

    def test_no_private_anntorchdataset(self):
        """scvi.dataloaders._anntorchdataset must not be imported."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from":
                assert not module.startswith("scvi.dataloaders._anntorchdataset"), (
                    "Found private 'scvi.dataloaders._anntorchdataset' — "
                    "must use public 'from scvi.data import AnnTorchDataset'"
                )

    def test_public_anntorchdataset(self):
        """from scvi.data import AnnTorchDataset must be present."""
        found = any(
            kind == "from" and module == "scvi.data" and "AnnTorchDataset" in names
            for kind, module, names in _collect_imports(self.PATH)
        )
        assert found, "Missing 'from scvi.data import AnnTorchDataset' in _manager.py"


# ---------------------------------------------------------------------------
# Step 3 — _multi_datasplitter.py: lightning 2, no parse_use_gpu_arg, no use_gpu
# ---------------------------------------------------------------------------

class TestDataSplitter:
    PATH = SRC / "data" / "_multi_datasplitter.py"

    def test_no_pytorch_lightning(self):
        """pytorch_lightning must not be imported (replaced by lightning.pytorch)."""
        for kind, module, _ in _collect_imports(self.PATH):
            assert not module.startswith("pytorch_lightning"), (
                "Found 'pytorch_lightning' import — must use 'lightning.pytorch'"
            )

    def test_lightning_pytorch_imported(self):
        """lightning.pytorch must be imported."""
        found = any(
            module.startswith("lightning.pytorch") or module.startswith("lightning")
            for _, module, _ in _collect_imports(self.PATH)
        )
        assert found, "Missing 'lightning.pytorch' import in _multi_datasplitter.py"

    def test_no_parse_use_gpu_arg(self):
        """parse_use_gpu_arg (removed in scvi 1.1) must not be imported."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from":
                assert "parse_use_gpu_arg" not in names, (
                    "Found 'parse_use_gpu_arg' import — removed in scvi-tools 1.1"
                )

    def test_no_anndata_import(self):
        """'from anndata import AnnData' must be removed (annotation fixed to AnnDataManager)."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from" and module == "anndata":
                assert "AnnData" not in names, (
                    "Found 'from anndata import AnnData' — "
                    "parameter annotation must use AnnDataManager"
                )

    def test_no_use_gpu_scvi_model_utils(self):
        """scvi.model._utils must not be imported (parse_use_gpu_arg lived there)."""
        for kind, module, _ in _collect_imports(self.PATH):
            assert not module.startswith("scvi.model._utils"), (
                "Found import from 'scvi.model._utils' — must be removed"
            )

    def test_anndatamanager_annotation(self):
        """AnnDataManager must be imported for the updated type annotation."""
        found = any(
            "AnnDataManager" in names
            for _, _, names in _collect_imports(self.PATH)
        )
        assert found, "Missing AnnDataManager import in _multi_datasplitter.py"

    def test_torch_imported(self):
        """torch must be imported (used for cuda.is_available())."""
        found = any(
            (kind == "import" and module == "torch") or
            (kind == "from" and module == "torch")
            for kind, module, _ in _collect_imports(self.PATH)
        )
        assert found, "Missing 'import torch' in _multi_datasplitter.py"


# ---------------------------------------------------------------------------
# Step 4 — training_mixin.py: use_gpu removed, Union removed
# ---------------------------------------------------------------------------

class TestTrainingMixin:
    PATH = SRC / "model" / "base" / "training_mixin.py"

    def test_no_use_gpu_in_train_signature(self):
        """use_gpu parameter must be removed from train()."""
        tree = ast.parse(self.PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train":
                arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
                assert "use_gpu" not in arg_names, (
                    "train() still has 'use_gpu' parameter — must be removed"
                )
                return
        pytest.fail("train() function not found in training_mixin.py")

    def test_no_union_import(self):
        """Union must not be imported (only used with use_gpu, which is removed)."""
        for kind, module, names in _collect_imports(self.PATH):
            if kind == "from" and module == "typing":
                assert "Union" not in names, (
                    "Found 'Union' in typing imports — unused after use_gpu removal"
                )

    def test_no_use_gpu_in_datasplitter_call(self):
        """use_gpu= must not appear in MultiGroupDataSplitter(...) call."""
        source = self.PATH.read_text()
        # Simple text search is sufficient; AST keyword detection is equivalent
        assert "use_gpu=use_gpu" not in source, (
            "Found 'use_gpu=use_gpu' in _multi_datasplitter call — must be removed"
        )

    def test_no_use_gpu_in_trainrunner_call(self):
        """use_gpu= must not appear in TrainRunner(...) call."""
        source = self.PATH.read_text()
        assert "use_gpu=" not in source, (
            "Found 'use_gpu=' in TrainRunner call — kwarg removed in scvi-tools 1.x"
        )


# ---------------------------------------------------------------------------
# Step 5 — pyproject.toml: version pins updated
# ---------------------------------------------------------------------------

class TestPyprojectToml:
    PATH = pathlib.Path(__file__).parent.parent / "pyproject.toml"

    def _text(self):
        return self.PATH.read_text()

    def test_scvi_tools_pin_updated(self):
        """scvi-tools must be pinned to >=1.0,<2 (not <0.21)."""
        text = self._text()
        assert "scvi-tools>=0.20" not in text, (
            "pyproject.toml still pins scvi-tools<0.21 — must be updated to >=1.0,<2"
        )
        assert "scvi-tools>=1.0" in text, (
            "pyproject.toml must pin 'scvi-tools>=1.0,<2'"
        )

    def test_requires_python_updated(self):
        """requires-python must not cap at <3.12."""
        text = self._text()
        assert "<3.12" not in text, (
            "pyproject.toml caps Python at <3.12 — must be removed (use >=3.10)"
        )
        assert ">=3.10" in text, (
            "pyproject.toml must set requires-python = '>=3.10'"
        )


# ---------------------------------------------------------------------------
# Integration tests (require scvi-tools>=1.0 installed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Runtime checks that require scvi 1.x to be installed."""

    def test_scvi_version(self):
        import scvi
        major = int(scvi.__version__.split(".")[0])
        assert major >= 1, f"Expected scvi-tools>=1.0, got {scvi.__version__}"

    def test_anntorchdataset_importable_from_scvi_data(self):
        from scvi.data import AnnTorchDataset  # noqa: F401

    def test_lightning_pytorch_importable(self):
        import lightning.pytorch  # noqa: F401

    def test_no_parse_use_gpu_arg_in_scvi(self):
        with pytest.raises(ImportError):
            from scvi.model._utils import parse_use_gpu_arg  # noqa: F401

    def test_spvipes_importable(self):
        import spVIPES  # noqa: F401
