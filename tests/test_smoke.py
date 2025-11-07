import importlib

import pytest


def test_core_packages_importable():
    """Ensure primary ORIGAMI packages can be imported."""
    assert importlib.import_module("modules") is not None
    assert importlib.import_module("models.origami") is not None
    assert importlib.import_module("utils") is not None


def test_pyrosetta_optional():
    """Skip gracefully when PyRosetta is not installed locally."""
    pytest.importorskip("pyrosetta", reason="PyRosetta requires a separate academic license")

