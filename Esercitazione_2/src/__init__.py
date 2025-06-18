"""
Main module for the Esercitazione_2 project - Deep Reinforcement Learning
"""

# Import main modules
from . import cartpole
from . import CarRacing
from . import common

# Import main functions for convenience
from .cartpole import (
    Policy, ValueNetwork,
    ReinforceTrainer, ReinforceWithBaselineTrainer,
    evaluate_agent, record_video,
    get_cartpole_hyperparameters, get_lunar_lander_hyperparameters, get_device
)

from .common import (
    moving_average, plot_training_scores, plot_comparison_scores,
    plot_carracing_scores, analyze_training_results, episodes_to_target,
    plot_cartpole_comparison, carracing_reinforce_training,
    display_video, display_carracing_video, display_cartpole_video
)

from .CarRacing import (
    Agent, Env, configure, RunManager,
    record_carracing_video, evaluate_carracing_agent
)

__all__ = [
    # Cartpole
    'Policy', 'ValueNetwork', 'ReinforceTrainer', 'ReinforceWithBaselineTrainer',
    'evaluate_agent', 'record_video', 'get_cartpole_hyperparameters', 
    'get_lunar_lander_hyperparameters', 'get_device',
    
    # Common analysis and training
    'moving_average', 'plot_training_scores', 'plot_comparison_scores',
    'plot_carracing_scores', 'analyze_training_results', 'episodes_to_target',
    'plot_cartpole_comparison', 'carracing_reinforce_training',
    'display_video', 'display_carracing_video', 'display_cartpole_video',
    
    # CarRacing
    'Agent', 'Env', 'configure', 'RunManager',
    'record_carracing_video', 'evaluate_carracing_agent',
    
    # Submodules
    'cartpole', 'CarRacing', 'common'
]
