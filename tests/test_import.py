"""Basic import tests for LatinX package."""


def test_import_latinx():
    """Test that the main latinx package can be imported."""
    import latinx

    assert latinx.__version__ == "0.1.0"


def test_import_submodules():
    """Test that submodules can be imported."""
    from latinx import data, models

    assert data is not None
    assert models is not None


def test_dependencies():
    """Test that key dependencies can be imported."""
    import jax
    import numpy as np
    import pandas as pd
    import torch

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert jax.__version__ is not None
    assert torch.__version__ is not None
