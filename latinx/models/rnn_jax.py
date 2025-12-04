"""
Simple RNN implementation in JAX with the same architecture as the PyTorch version.

This module provides a JAX/Flax implementation of SimpleRNN that matches
the architecture and behavior of the PyTorch version in rnn.py.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import Array


class SimpleRNN(nn.Module):
    """
    Simple RNN model with feature extraction using JAX/Flax.

    This implementation matches the PyTorch version with:
    - Single RNN layer with tanh nonlinearity
    - Batch-first input format
    - Returns last step features and final hidden state

    Args:
        input_size: Dimension of input features (default: 1)
        hidden_size: Dimension of hidden state (default: 32)

    Example:
        >>> import jax.random as random
        >>> model = SimpleRNN(input_size=1, hidden_size=32)
        >>> key = random.PRNGKey(0)
        >>> batch_size, seq_len = 16, 10
        >>> x = jnp.ones((batch_size, seq_len, 1))
        >>> variables = model.init(key, x)
        >>> features, h_n = model.apply(variables, x)
        >>> print(features.shape)  # (batch_size, hidden_size)
    """

    input_size: int = 1
    hidden_size: int = 32

    @nn.compact
    def __call__(
        self, x: Array, h: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        """
        Forward pass returning backbone features only (no prediction head).

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            h: Optional initial hidden state of shape (batch, hidden_size)
               If None, initializes to zeros.

        Returns:
            Tuple of:
                - last_step_features: Features from last time step, shape (batch, hidden_size)
                - h_n: Final hidden state, shape (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if h is None:
            h = jnp.zeros((batch_size, self.hidden_size))

        # RNN cell weights
        # Weight for input: (hidden_size, input_size)
        # Weight for hidden: (hidden_size, hidden_size)
        # Bias: (hidden_size,)
        W_ih = self.param(
            "W_ih",
            nn.initializers.xavier_uniform(),
            (self.hidden_size, self.input_size),
        )
        W_hh = self.param(
            "W_hh",
            nn.initializers.xavier_uniform(),
            (self.hidden_size, self.hidden_size),
        )
        b_ih = self.param("b_ih", nn.initializers.zeros, (self.hidden_size,))
        b_hh = self.param("b_hh", nn.initializers.zeros, (self.hidden_size,))

        # Process sequence step by step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            # RNN update: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
            h = jnp.tanh(
                x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            )
            outputs.append(h)

        # Stack outputs: (seq_len, batch, hidden_size) -> (batch, seq_len, hidden_size)
        out = jnp.stack(outputs, axis=1)

        # Extract last step features
        last_step_features = out[:, -1, :]  # (batch, hidden_size)

        return last_step_features, h


class PredictionHead(nn.Module):
    """
    Simple linear prediction head for training the RNN.

    This is a temporary module used during RNN pre-training to provide
    supervised learning targets. After training, this head is discarded
    and only the RNN features are used with Bayesian or Kalman filter heads.

    Args:
        output_dim: Output dimension (default: 1 for scalar prediction)

    Example:
        >>> # Create RNN and prediction head
        >>> rnn_model, rnn_params = create_rnn(hidden_size=32)
        >>> pred_head, head_params = create_prediction_head(hidden_size=32)
        >>>
        >>> # Forward pass during training
        >>> features, _ = rnn_model.apply(rnn_params, x)
        >>> prediction = pred_head.apply(head_params, features)
    """

    output_dim: int = 1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through linear layer.

        Args:
            x: Input features of shape (batch, feature_dim)

        Returns:
            Predictions of shape (batch, output_dim) or (batch,) if output_dim=1
        """
        out = nn.Dense(self.output_dim)(x)
        if self.output_dim == 1:
            return out.squeeze(-1)  # (batch,) for scalar output
        return out


def create_rnn(
    input_size: int = 1,
    hidden_size: int = 32,
    seed: int = 0,
) -> Tuple[SimpleRNN, Dict[str, Any]]:
    """
    Convenience function to create and initialize an RNN model.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        seed: Random seed for initialization

    Returns:
        Tuple of (model, initial_params) where initial_params contains
        the initialized model parameters.

    Example:
        >>> model, params = create_rnn(input_size=1, hidden_size=32, seed=42)
        >>> x = jnp.ones((8, 10, 1))  # (batch=8, seq_len=10, features=1)
        >>> features, h_n = model.apply(params, x)
    """
    import jax.random as random

    model = SimpleRNN(input_size=input_size, hidden_size=hidden_size)
    key = random.PRNGKey(seed)

    # Initialize with dummy input
    dummy_x = jnp.ones((1, 1, input_size))
    params = model.init(key, dummy_x)

    return model, params # type: ignore


def create_prediction_head(
    hidden_size: int,
    output_dim: int = 1,
    seed: int = 0,
) -> Tuple[PredictionHead, Dict[str, Any]]:
    """
    Factory function to create a prediction head for RNN pre-training.

    This creates a simple linear layer that maps RNN features to predictions.
    Use this during pre-training, then discard it and use Bayesian/Kalman heads.

    Args:
        hidden_size: Feature dimension (should match RNN hidden_size)
        output_dim: Output dimension (default: 1 for scalar prediction)
        seed: Random seed for initialization

    Returns:
        Tuple of (head_model, initial_params) where initial_params contains
        the initialized model parameters.

    Example:
        >>> # Create RNN
        >>> rnn_model, rnn_params = create_rnn(hidden_size=32, seed=42)
        >>>
        >>> # Create prediction head
        >>> pred_head, head_params = create_prediction_head(hidden_size=32, seed=42)
        >>>
        >>> # Training loop
        >>> for x, y in train_data:
        >>>     features, _ = rnn_model.apply(rnn_params, x)
        >>>     predictions = pred_head.apply(head_params, features)
        >>>     loss = mse_loss(predictions, y)
        >>>     # Update both rnn_params and head_params
        >>>
        >>> # After training, discard head_params and use only rnn_params with BLL/KF
    """
    import jax.random as random

    # Create dummy features for initialization (values don't matter, only shape)
    dummy_features = jnp.ones((1, hidden_size))

    # Initialize head with dummy features
    head = PredictionHead(output_dim=output_dim)
    key = random.PRNGKey(seed)
    params = head.init(key, dummy_features)

    return head, params  # type: ignore


def create_loss_fn(
    rnn_model: SimpleRNN,
    pred_head: PredictionHead,
) -> Callable[[Dict[str, Any], Dict[str, Any], Array, Array], float]:
    """
    Create a loss function for training RNN with prediction head.

    This function returns a loss function that computes MSE loss between
    predictions and targets. The returned function can be used with
    jax.value_and_grad for backpropagation.

    Args:
        rnn_model: The RNN model for feature extraction
        pred_head: The prediction head model

    Returns:
        A loss function with signature:
            loss_fn(rnn_params, head_params, x, y_true) -> loss_value

    Example:
        >>> rnn_model, rnn_params = create_rnn(hidden_size=32)
        >>> pred_head, head_params = create_prediction_head(hidden_size=32)
        >>>
        >>> loss_fn = create_loss_fn(rnn_model, pred_head)
        >>> loss = loss_fn(rnn_params, head_params, x_batch, y_batch)
        >>>
        >>> # For backpropagation
        >>> loss_value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        ...     rnn_params, head_params, x_batch, y_batch
        ... )
    """

    def loss_fn(
        rnn_params: Dict[str, Any],
        head_params: Dict[str, Any],
        x: Array,
        y_true: Array,
    ) -> float:
        """
        Compute MSE loss for RNN + prediction head.

        Args:
            rnn_params: RNN parameters
            head_params: Prediction head parameters
            x: Input sequences of shape (batch, seq_len, input_size)
            y_true: Target values of shape (batch,)

        Returns:
            Scalar MSE loss value
        """
        # Forward pass through RNN
        features, _ = rnn_model.apply(rnn_params, x)

        # Forward pass through prediction head
        y_pred = pred_head.apply(head_params, features)

        # Compute MSE loss
        loss = jnp.mean((y_pred - y_true) ** 2)
        return loss

    return loss_fn


def create_train_step(
    rnn_model: SimpleRNN,
    pred_head: PredictionHead,
    rnn_optimizer: optax.GradientTransformation,
    head_optimizer: optax.GradientTransformation,
) -> Callable:
    """
    Create a JIT-compiled training step function.

    This function returns a training step that:
    1. Computes loss and gradients for both RNN and prediction head
    2. Updates parameters using the provided optimizers
    3. Updates optimizer states
    4. Returns updated parameters, states, and loss value

    Args:
        rnn_model: The RNN model for feature extraction
        pred_head: The prediction head model
        rnn_optimizer: Optimizer for RNN parameters (e.g., optax.adam(1e-3))
        head_optimizer: Optimizer for head parameters (e.g., optax.adam(1e-3))

    Returns:
        A JIT-compiled training step function with signature:
            train_step(rnn_params, head_params, rnn_opt_state, head_opt_state, x, y)
            -> (updated_rnn_params, updated_head_params, updated_rnn_state,
                updated_head_state, loss_value)

    Example:
        >>> # Create models
        >>> rnn_model, rnn_params = create_rnn(hidden_size=32, seed=42)
        >>> pred_head, head_params = create_prediction_head(hidden_size=32, seed=42)
        >>>
        >>> # Create optimizers
        >>> rnn_optimizer = optax.adam(1e-3)
        >>> head_optimizer = optax.adam(1e-3)
        >>> rnn_opt_state = rnn_optimizer.init(rnn_params)
        >>> head_opt_state = head_optimizer.init(head_params)
        >>>
        >>> # Create training step
        >>> train_step = create_train_step(rnn_model, pred_head, rnn_optimizer, head_optimizer)
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in train_data:
        >>>     rnn_params, head_params, rnn_opt_state, head_opt_state, loss = train_step(
        >>>         rnn_params, head_params, rnn_opt_state, head_opt_state, x_batch, y_batch
        >>>     )
        >>>     print(f"Loss: {loss:.6f}")
    """
    # Create loss function
    loss_fn = create_loss_fn(rnn_model, pred_head)

    @jax.jit
    def train_step(
        rnn_params: Dict[str, Any],
        head_params: Dict[str, Any],
        rnn_opt_state: optax.OptState,
        head_opt_state: optax.OptState,
        x: Array,
        y: Array,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], optax.OptState, optax.OptState, float]:
        """
        Single training step with backpropagation.

        Args:
            rnn_params: Current RNN parameters
            head_params: Current head parameters
            rnn_opt_state: Current RNN optimizer state
            head_opt_state: Current head optimizer state
            x: Input batch of shape (batch, seq_len, input_size)
            y: Target batch of shape (batch,)

        Returns:
            Tuple of (updated_rnn_params, updated_head_params, updated_rnn_state,
                     updated_head_state, loss_value)
        """
        # Compute loss and gradients for BOTH RNN and head
        loss_value, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            rnn_params, head_params, x, y
        )

        # grads is a tuple: (rnn_grads, head_grads)
        rnn_grads, head_grads = grads

        # Update RNN parameters
        rnn_updates, rnn_opt_state = rnn_optimizer.update(rnn_grads, rnn_opt_state)
        rnn_params = optax.apply_updates(rnn_params, rnn_updates)

        # Update head parameters
        head_updates, head_opt_state = head_optimizer.update(head_grads, head_opt_state)
        head_params = optax.apply_updates(head_params, head_updates)

        return rnn_params, head_params, rnn_opt_state, head_opt_state, loss_value

    return train_step
