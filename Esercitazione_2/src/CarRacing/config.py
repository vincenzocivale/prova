import os
import json
import torch
import numpy as np


class Args:
    def __init__(self, trial=5):
        # Experiment
        self.checkpoint = 0
        self.test = False
        self.trial = trial
        self.wandb_project = "CarRacing-v0-PPO"  
        self.start_episode = self.checkpoint

        # PPO settings
        self.gamma = 0.99
        self.clip_param = 0.4
        self.ppo_epoch = 10
        self.buffer_capacity = 500
        self.batch_size = 128

        # Env settings
        self.seed = 0
        self.action_repeat = 4
        self.deathThreshold = 2000
        self.deathByGreeneryThreshold = 35
        self.maxDistance = 100
        self.max_episode_steps = 10000  
        
        # Observation/action stacking
        self.valueStackSize = 8
        self.actionStack = 0
        self.numberOfLasers = 5

        # Action transformation
        self.actionMultiplier = np.array([2.0, 1.0, 1.0])
        self.actionBias = np.array([-1.0, 0.0, 0.0])

        # Save path
        self.saveLocation = f"model/distances/train_{trial}/"
        os.makedirs(self.saveLocation, exist_ok=True)

    def getParamsDict(self):
        ret = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
        print("\nHYPERPARAMETERS =", ret)
        return ret

    def actionTransformation(self, action):
        return action * self.actionMultiplier + self.actionBias


def configure():
    args = Args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Salva parametri come JSON leggibile
    with open(os.path.join(args.saveLocation, 'params.json'), 'w') as f:
        json.dump(args.getParamsDict(), f, indent=4)

    return args, use_cuda, device
