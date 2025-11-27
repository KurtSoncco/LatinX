import jax.numpy as jnp
import pytest

from latinx.models.kalman_filter import KalmanFilterHead


@pytest.fixture
def test_phi_x():
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


def test_predict(test_phi_x):
    kf = KalmanFilterHead(feature_dim=10, rho=0.99, Q_std=0.01, R_std=0.1)
    y_pred_val = kf.predict(test_phi_x)
    assert jnp.isclose(y_pred_val, 0.0, atol=1e-6)


def test_update(test_phi_x):
    kf = KalmanFilterHead(feature_dim=10, rho=0.99, Q_std=0.01, R_std=0.1)
    y_true = 10.0
    y_pred_val = kf.predict(test_phi_x)
    innovation_covariance, prediction_error = kf.update(y_true, y_pred_val)
    assert jnp.isclose(innovation_covariance, 377.38699999999994, atol=1e-6)
    assert jnp.isclose(prediction_error, 10.0, atol=1e-6)
