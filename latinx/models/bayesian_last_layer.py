"""
Neural network with Bayesian last layer implementation.

This module implements a plugin approach where a standard neural network is trained,
then the last layer is removed and replaced with a Bayesian linear regression model.
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from copy import deepcopy
from typing import Callable, Dict, Any, Tuple


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""

    hidden_dims: Tuple[int, ...] = (10, 10)
    output_dim: int = 1
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim, name="last-layer")(x)
        return x


class MLPHidden(nn.Module):
    """Multi-layer perceptron without the last layer (feature extractor)."""

    hidden_dims: Tuple[int, ...] = (10, 10)
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.activation(x)
        return x


class BayesianLastLayer:
    """
    Neural network with Bayesian last layer using plugin approach.

    This class trains a standard neural network, then removes the last layer
    and fits a Bayesian linear regression model on top of the learned features.

    Args:
        hidden_dims: Tuple of hidden layer dimensions.
        sigma: Observation noise standard deviation.
        alpha: Prior precision parameter for Bayesian regression.
        learning_rate: Learning rate for neural network training.
        n_steps: Number of training steps.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (10, 10),
        sigma: float = 0.3,
        alpha: float = 0.05,
        learning_rate: float = 1e-3,
        n_steps: int = 3000,
        seed: int = 31415,
    ):
        self.hidden_dims = hidden_dims
        self.sigma = sigma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.seed = seed

        # Models
        self.model = MLP(hidden_dims=hidden_dims)
        self.model_hidden = MLPHidden(hidden_dims=hidden_dims)

        # Parameters (initialized after training)
        self.params_full = None
        self.params_hidden = None

        # Bayesian last layer parameters
        self.posterior_mean = None
        self.posterior_precision = None
        self.beta = None  # observation precision

    def _lossfn(self, params: Dict, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """Mean squared error loss function."""
        yhat = self.model.apply(params, x).squeeze()
        return jnp.power(y.squeeze() - yhat, 2).mean()

    def _step(self, state: TrainState, _: Any, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[TrainState, float]:
        """Single training step."""
        loss, grads = jax.value_and_grad(lambda p: self._lossfn(p, x, y))(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def _train_neural_network(self, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[Dict, jnp.ndarray]:
        """
        Train the full neural network.

        Args:
            x: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1) or (n_samples,).

        Returns:
            Tuple of (trained_parameters, loss_history).
        """
        key = jax.random.PRNGKey(self.seed)
        params_init = self.model.init(key, x)

        state = TrainState.create(
            params=params_init,
            apply_fn=self.model.apply,
            tx=optax.adam(self.learning_rate)
        )

        # Training loop
        step_fn = lambda state, i: self._step(state, i, x, y)
        state_final, loss_history = jax.lax.scan(step_fn, state, jnp.arange(self.n_steps))

        return state_final.params, loss_history

    def _extract_hidden_params(self, params_full: Dict) -> Dict:
        """Extract parameters for all layers except the last one."""
        params_hidden = deepcopy(params_full)
        del params_hidden["params"]["last-layer"]
        return params_hidden

    def _fit_bayesian_last_layer(self, x: jnp.ndarray, y: jnp.ndarray):
        """
        Fit Bayesian linear regression on the learned features.

        Args:
            x: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1) or (n_samples,).
        """
        # Extract features using the hidden layers
        phi_x = self.model_hidden.apply(self.params_hidden, x)

        M = phi_x.shape[1]
        IM = jnp.eye(M)
        self.beta = 1 / self.sigma ** 2  # observation precision

        # Compute posterior parameters
        self.posterior_precision = self.alpha * IM + self.beta * phi_x.T @ phi_x
        self.posterior_mean = self.beta * jnp.linalg.solve(self.posterior_precision, phi_x.T @ y)
        self.posterior_mean = self.posterior_mean.ravel()

    def fit(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the Bayesian last layer model.

        This method:
        1. Trains the full neural network
        2. Extracts the hidden layer parameters
        3. Fits a Bayesian linear regression on the learned features

        Args:
            x: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1) or (n_samples,).

        Returns:
            Training loss history.
        """
        # Ensure y has shape (n_samples, 1)
        if y.ndim == 1:
            y = y[:, None]

        # Train the full neural network
        self.params_full, loss_history = self._train_neural_network(x, y)

        # Extract hidden layer parameters
        self.params_hidden = self._extract_hidden_params(self.params_full)

        # Fit Bayesian last layer
        self._fit_bayesian_last_layer(x, y)

        return loss_history

    def predict(self, x: jnp.ndarray, return_std: bool = False) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions using the Bayesian last layer.

        Args:
            x: Input features of shape (n_samples, n_features).
            return_std: If True, return both mean and standard deviation.

        Returns:
            If return_std=False: predictions of shape (n_samples,).
            If return_std=True: tuple of (predictions, std) both of shape (n_samples,).
        """
        if self.posterior_mean is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        # Extract features
        phi_x = self.model_hidden.apply(self.params_hidden, x)

        # Compute predictive mean
        pred_mean = jnp.einsum("d,td->t", self.posterior_mean, phi_x)

        if return_std:
            # Compute predictive variance
            pred_var = 1 / self.beta + jnp.einsum(
                "si,ij,sj->s",
                phi_x,
                jnp.linalg.inv(self.posterior_precision),
                phi_x
            )
            pred_std = jnp.sqrt(pred_var)
            return pred_mean, pred_std

        return pred_mean

    def predict_full_nn(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the original trained neural network (before Bayesian last layer).

        Args:
            x: Input features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        if self.params_full is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        return self.model.apply(self.params_full, x).squeeze()
