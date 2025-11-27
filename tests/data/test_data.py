"""Tests for data processing utilities."""

import numpy as np


def test_generate_sine_cosine_data():
    """Test sine and cosine data generation."""
    from latinx.data.sine_cosine import SineCosineTranslator

    translator = SineCosineTranslator(amplitude=2, angle_multiplier=np.pi, seed=42, num_samples=10)
    data = translator.generate()

    assert len(data) == 10
    assert ["t", "sine", "cosine", "x1", "y1"] == list(data.columns)
    assert np.isclose(np.max(data["sine"]), 1.6249060130820974, atol=1e-5)
    assert np.isclose(np.min(data["cosine"]), -1.7983789130292258, atol=1e-5)
