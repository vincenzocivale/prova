import torch.nn as nn
from config import Args

class Net(nn.Module):
    """
    Actor-Critic Network for PPO in CarRacing.
    """

    def __init__(self, args: Args):
        super().__init__()
        input_dim = args.valueStackSize * args.numberOfLasers + 3 * args.actionStack

        # Base fully connected layer: input -> 128 features
        self.cnn_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

        # Value head: predicts state value (critic)
        self.v = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # Shared actor features
        self.fc = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU()
        )

        # Actor output heads: alpha and beta parameters for Beta distribution
        self.alpha_head = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        # Initialize weights for Linear layers (Conv2d is not used here)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): input tensor with shape (batch_size, input_dim)

        Returns:
            (tuple): tuple containing
                - (alpha, beta): parameters of Beta distributions (for stochastic policy)
                - v: state value estimate
        """
        x = self.cnn_base(x)       # [batch_size, 128]
        x = x.view(-1, 128)       # flatten if necessary (usually batch dimension remains)
        v = self.v(x)             # value head output, shape [batch_size, 1]
        actor_feat = self.fc(x)   # actor feature embedding

        alpha = self.alpha_head(actor_feat) + 1  # shift to > 1 for stability
        beta = self.beta_head(actor_feat) + 1

        return (alpha, beta), v


