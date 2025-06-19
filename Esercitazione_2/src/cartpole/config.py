"""
Configuration and hyperparameters for REINFORCE training
"""

def get_cartpole_hyperparameters():
    """Returns hyperparameters for CartPole-v1"""
    return {
        "h_size": 16,
        "n_training_episodes": 600,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "env_id": "CartPole-v1",
        "print_every": 10,
        "save_path": "best_reinforce.pt"
    }


def get_lunar_lander_hyperparameters():
    """Returns hyperparameters for LunarLander-v2"""
    return {
        "h_size": 64,
        "n_training_episodes": 2500,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 0.99,
        "lr": 5e-3,
        "env_id": "LunarLander-v2",
        "print_every": 50,
        "save_path": "best_lunar_lander.pt"
    }


def get_device():
    """Returns the device for PyTorch (GPU if available, otherwise CPU)"""
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
