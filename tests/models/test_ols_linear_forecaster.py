"""Tests for OLSLinearForecaster."""

import numpy as np
import pytest

from latinx.models.ols_linear_forecaster import (
    OLSLinearForecaster,
    OLSLinearForecasterConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    return OLSLinearForecasterConfig(L=32, H=8)


@pytest.fixture
def cfg_no_norm():
    return OLSLinearForecasterConfig(L=32, H=8, instance_norm=False)


@pytest.fixture
def linear_data():
    """y = 2*mean(x) + noise  — simple relationship for sanity check."""
    rng = np.random.default_rng(42)
    N, L, H = 200, 32, 8
    X = rng.standard_normal((N, L))
    means = X.mean(axis=1, keepdims=True)
    Y = np.broadcast_to(2 * means, (N, H)) + rng.normal(0, 0.01, (N, H))
    return X, Y


# ── Initialisation ────────────────────────────────────────────────────


def test_config_defaults():
    cfg = OLSLinearForecasterConfig(L=64, H=16)
    assert cfg.alpha == 1e-6
    assert cfg.instance_norm is True
    assert cfg.epsilon == 1e-5


def test_unfitted_state(cfg):
    model = OLSLinearForecaster(cfg)
    assert model.W is None
    assert model._fitted is False


# ── Predict before fit ────────────────────────────────────────────────


def test_predict_before_fit_raises(cfg):
    model = OLSLinearForecaster(cfg)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(cfg.L))


def test_predict_batch_before_fit_raises(cfg):
    model = OLSLinearForecaster(cfg)
    with pytest.raises(RuntimeError):
        model.predict_batch(np.zeros((5, cfg.L)))


# ── Shape checks ──────────────────────────────────────────────────────


def test_weight_shape(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)
    assert model.W.shape == (cfg.L + 1, cfg.H)  # instance_norm adds 1


def test_weight_shape_no_norm(cfg_no_norm, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg_no_norm).fit(X, Y)
    assert model.W.shape == (cfg_no_norm.L, cfg_no_norm.H)


def test_predict_output_shape(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)
    y_hat = model.predict(X[0])
    assert y_hat.shape == (cfg.H,)


def test_predict_batch_output_shape(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)
    Y_hat = model.predict_batch(X[:10])
    assert Y_hat.shape == (10, cfg.H)


# ── Dimension mismatch ───────────────────────────────────────────────


def test_fit_wrong_context_raises(cfg):
    model = OLSLinearForecaster(cfg)
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, cfg.L + 5)), np.zeros((10, cfg.H)))


def test_fit_wrong_horizon_raises(cfg):
    model = OLSLinearForecaster(cfg)
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, cfg.L)), np.zeros((10, cfg.H + 3)))


def test_predict_wrong_length_raises(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)
    with pytest.raises(ValueError):
        model.predict(np.zeros(cfg.L + 1))


def test_predict_batch_wrong_length_raises(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)
    with pytest.raises(ValueError):
        model.predict_batch(np.zeros((5, cfg.L + 1)))


# ── Instance norm equivariance ────────────────────────────────────────


def test_instance_norm_shift_equivariance(cfg, linear_data):
    """Shifting input by a constant should shift the output by that constant."""
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)

    x = X[0]
    y_base = model.predict(x)

    shift = 42.0
    y_shifted = model.predict(x + shift)

    np.testing.assert_allclose(y_shifted, y_base + shift, atol=1e-6)


# ── Works without instance norm ──────────────────────────────────────


def test_fit_predict_no_instance_norm(cfg_no_norm, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg_no_norm).fit(X, Y)
    y_hat = model.predict(X[0])
    assert y_hat.shape == (cfg_no_norm.H,)
    assert np.all(np.isfinite(y_hat))


# ── Regularisation effect ────────────────────────────────────────────


def test_higher_alpha_smaller_weights(linear_data):
    X, Y = linear_data

    cfg_low = OLSLinearForecasterConfig(L=32, H=8, alpha=1e-8)
    cfg_high = OLSLinearForecasterConfig(L=32, H=8, alpha=10.0)

    m_low = OLSLinearForecaster(cfg_low).fit(X, Y)
    m_high = OLSLinearForecaster(cfg_high).fit(X, Y)

    norm_low = float(np.linalg.norm(m_low.W))
    norm_high = float(np.linalg.norm(m_high.W))

    assert norm_low > norm_high


# ── Correctness on simple synthetic data ──────────────────────────────


def test_correct_fit_identity():
    """If Y = X (identity mapping, L==H), weights ≈ I (centered)."""
    rng = np.random.default_rng(0)
    L = 16
    N = 200
    X = rng.standard_normal((N, L))
    Y = X.copy()

    cfg = OLSLinearForecasterConfig(L=L, H=L, alpha=1e-10, instance_norm=False)
    model = OLSLinearForecaster(cfg).fit(X, Y)

    # Y_centered = X_centered, so W should be close to identity
    np.testing.assert_allclose(model.W, np.eye(L), atol=1e-4)


def test_predict_close_to_target(linear_data):
    """Predictions should be close to training targets for clean data."""
    X, Y = linear_data
    cfg = OLSLinearForecasterConfig(L=32, H=8, alpha=1e-10)
    model = OLSLinearForecaster(cfg).fit(X, Y)

    Y_hat = model.predict_batch(X)
    mse = float(np.mean((Y_hat - Y) ** 2))
    assert mse < 0.01  # very clean data, should fit well


# ── Method chaining ──────────────────────────────────────────────────


def test_fit_returns_self(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg)
    result = model.fit(X, Y)
    assert result is model


# ── predict vs predict_batch consistency ──────────────────────────────


def test_predict_batch_matches_predict(cfg, linear_data):
    X, Y = linear_data
    model = OLSLinearForecaster(cfg).fit(X, Y)

    Y_batch = model.predict_batch(X[:5])
    for i in range(5):
        y_single = model.predict(X[i])
        np.testing.assert_allclose(Y_batch[i], y_single, atol=1e-10)
