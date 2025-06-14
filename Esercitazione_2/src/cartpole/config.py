"""
Configurazione e iperparametri per il training REINFORCE
"""

def get_cartpole_hyperparameters():
    """Restituisce gli iperparametri per CartPole-v1"""
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
    """Restituisce gli iperparametri per LunarLander-v2"""
    return {
        "h_size": 64,
        "n_training_episodes": 2000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 0.99,
        "lr": 5e-3,
        "env_id": "LunarLander-v2",
        "print_every": 50,
        "save_path": "best_lunar_lander.pt"
    }


def get_device():
    """Restituisce il device per PyTorch (GPU se disponibile, altrimenti CPU)"""
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
