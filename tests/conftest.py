"""Shared test fixtures and configuration."""

import sys
from pathlib import Path

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parent.parent
