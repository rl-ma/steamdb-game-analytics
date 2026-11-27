"""Unit tests verifying CI workflow and repository tooling configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_version_file_is_pinned() -> None:
    version = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert version == "3.13.3"


def test_pyproject_configures_ruff_and_pytest() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    ruff_cfg = config["tool"]["ruff"]
    assert ruff_cfg["target-version"] == "py313"
    assert "*.ipynb" in ruff_cfg["exclude"]

    pytest_cfg = config["tool"]["pytest"]["ini_options"]
    assert pytest_cfg["testpaths"] == ["tests"]
    assert any(m.startswith("integration:") for m in pytest_cfg["markers"])


def test_requirements_list_core_and_dev_dependencies() -> None:
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib"):
        assert pkg in req

    req_dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in req_dev
    assert "pytest" in req_dev
    assert "ruff" in req_dev


def test_ci_workflow_defines_lint_and_test_jobs() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert 'python-version: "3.13.3"' in workflow
    assert "ruff check ." in workflow
    assert "ruff format --check ." in workflow
    assert "pytest -m" in workflow


def test_gitignore_excludes_large_data_and_secrets() -> None:
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for pattern in ("*.parquet", "*.csv", ".env", ".venv/"):
        assert pattern in lines
