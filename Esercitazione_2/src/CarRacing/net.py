import torch.nn as nn
from .config import Args, configure

class Net(nn.Module):
    """
    Actor-Critic Network for PPO (discrete actions)
    """
    def __init__(self, args:Args):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential( 
            nn.Linear(args.valueStackSize*args.numberOfLasers + 3*args.actionStack, 128),
            nn.ReLU(),
        )
        self.v = nn.Sequential(nn.Linear(128, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(128, 100), nn.ReLU())
        # Nuova testa per logits discrete (5 azioni)
        self.logits_head = nn.Linear(100, 5)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)  # (batch_size, 128)
        x = x.view(-1, 128)
        v = self.v(x)
        x = self.fc(x)        # ora x è (batch_size, 100)
        logits = self.logits_head(x)  # (batch_size, 5)
        return logits, v



if __name__ == "__main__":
    print(Net(configure()[0]))
