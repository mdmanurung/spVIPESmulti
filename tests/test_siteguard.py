"""Regression tests for user-site package isolation."""

from __future__ import annotations

import importlib.util
import json
import os
import site
import subprocess
import sys
from pathlib import Path


def _load_siteguard_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "spVIPESmulti" / "_siteguard.py"
    spec = importlib.util.spec_from_file_location("_spvipesmulti_siteguard_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_guard_removes_user_site_path(monkeypatch, tmp_path):
    siteguard = _load_siteguard_module()

    fake_user_site = tmp_path / "user-site"
    fake_user_site.mkdir()

    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.delenv("SPVIPESMULTI_ALLOW_USER_SITE", raising=False)
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(fake_user_site))
    monkeypatch.setattr(site, "getuserbase", lambda: str(tmp_path))
    monkeypatch.syspath_prepend(str(fake_user_site))

    removed = siteguard.apply_user_site_guard(fail_on_loaded=False)

    assert str(fake_user_site) in removed
    assert str(fake_user_site) not in sys.path
    assert site.ENABLE_USER_SITE is False
    assert os.environ["PYTHONNOUSERSITE"] == "1"


def test_sitecustomize_disables_user_site_for_src_startup(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    user_base = tmp_path / "userbase"
    user_site = user_base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    user_site.mkdir(parents=True)

    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("SPVIPESMULTI_ALLOW_USER_SITE", None)
    env["PYTHONUSERBASE"] = str(user_base)
    env["PYTHONPATH"] = str(src_path)

    code = """
import json
import os
import site
import sys

user_site = site.getusersitepackages()
print(json.dumps({
    "enable_user_site": site.ENABLE_USER_SITE,
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    "user_site_in_path": user_site in sys.path,
}))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True)
    payload = json.loads(result.stdout)

    assert payload == {
        "enable_user_site": False,
        "python_no_user_site": "1",
        "user_site_in_path": False,
    }


def test_sitecustomize_normalizes_inherited_conda_prefix():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)
    env["CONDA_PREFIX"] = "/definitely/not/the/running/env"
    env["CONDA_DEFAULT_ENV"] = "wrong-env"

    code = """
import json
import os
import sys
from pathlib import Path

is_conda = (Path(sys.prefix) / "conda-meta").is_dir()
print(json.dumps({
    "is_conda": is_conda,
    "prefix": sys.prefix,
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
}))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True)
    payload = json.loads(result.stdout)

    if payload["is_conda"]:
        assert payload["conda_prefix"] == payload["prefix"]
        assert payload["conda_default_env"] == Path(payload["prefix"]).name
    else:
        assert payload["conda_prefix"] == "/definitely/not/the/running/env"
