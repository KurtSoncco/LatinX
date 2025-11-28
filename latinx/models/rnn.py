import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    Simple RNN model with feature extraction and optional prediction head.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning backbone features only (no prediction head).

        Args:
            x: (batch, seq_len, input_size)
            h: Optional initial hidden state (1, batch, hidden_size)

        Returns:
            Tuple (last_step_features, h_n)
        """
        out, h_n = self.rnn(x, h)
        last_step_features = out[:, -1, :]
        return last_step_features, h_n