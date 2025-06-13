import torch.nn as nn
from .config import Args, configure

class Net(nn.Module):
    """
    Actor-Critic Network for PPO (discrete actions)
    """
    def __init__(self, args:Args):
        super(Net, self).__init__()
        input_dim = args.valueStackSize * args.numberOfLasers + 3 * args.actionStack

        self.base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Value head
        self.v = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(64, 5)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.base(x)  # (batch_size, 128)
        v = self.v(x)
        x = self.fc(x)    # (batch_size, 64)
        logits = self.logits_head(x)  # (batch_size, 5)
        return logits, v

if __name__ == "__main__":
    print(Net(configure()[0]))