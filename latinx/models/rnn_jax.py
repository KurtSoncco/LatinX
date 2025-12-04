"""
Simple RNN implementation in JAX with the same architecture as the PyTorch version.

This module provides a JAX/Flax implementation of SimpleRNN that matches
the architecture and behavior of the PyTorch version in rnn.py.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
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
