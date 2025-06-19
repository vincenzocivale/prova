"""
Training module for shared training functions
"""
import numpy as np
import torch
from collections import deque


def carracing_reinforce_training(agent, env, configs, run_manager, num_episodes=100, print_every=10):
    """
    REINFORCE-style training function for CarRacing
    Similar to CartPole training structure but adapted for CarRacing PPO
    
    Args:
        agent: The CarRacing agent to train
        env: The CarRacing environment
        configs: Experiment configurations
        run_manager: Manager for logging and saving
        num_episodes: Number of training episodes
        print_every: Frequency of result printing
    
    Returns:
        List of scores per episode
    """
    scores = []
    scores_deque = deque(maxlen=100)
    best_score = -float('inf')
    
    print(f"🚗 Starting training for {num_episodes} episodes...")
    
    for episode in range(configs.checkpoint, configs.checkpoint + num_episodes):
        episode_score = 0
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]  # Action history for environment
        
        # Execute a complete episode
        for t in range(configs.max_episode_steps):
            # Select action through policy
            action, log_prob = agent.select_action(state)
            
            # Execute step in environment
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            # Update agent buffer (similar to REINFORCE but with PPO buffer)
            if not configs.test:
                loss, entropy = agent.update((state, action, log_prob, reward, next_state), episode)
            
            episode_score += reward
            state = next_state
            action_history = [action_history[-1], action]
            
            if done:
                break
        
        # Results tracking
        scores.append(episode_score)
        scores_deque.append(episode_score)
        avg_score = np.mean(scores_deque)
        
        # Log metrics
        run_manager.log_episode(episode, episode_score)
        
        # Save best model
        if avg_score > best_score:
            best_score = avg_score
            run_manager.save_model("best_carracing_model.pth")
        
        # Periodic checkpoint
        run_manager.maybe_save_checkpoint(episode)
        
        # Print progress
        if episode % print_every == 0:
            print(f'Episode {episode}\tScore: {episode_score:.2f}\tAverage Score: {avg_score:.2f}')
            print(f'  Death reason: {death_reason}')
            print(f'  Best average: {best_score:.2f}')
        
        if configs.test:
            break
    
    return scores


def generic_episode_training(agent, env, max_steps=1000, device='cpu'):
    """
    Generic function to execute a training episode
    Usable for different types of agents and environments
    
    Args:
        agent: Agent with select_action and act methods
        env: Gymnasium-compatible environment
        max_steps: Maximum number of steps per episode
        device: Device for computations
    
    Returns:
        Tupla (rewards, log_probs, states, actions, total_reward)
    """
    rewards = []
    log_probs = []
    states = []
    actions = []
    
    # Reset environment
    state, _ = env.reset()
    total_reward = 0
    
    for t in range(max_steps):
        states.append(state)
        
        # Select action
        if hasattr(agent, 'act'):
            action, log_prob = agent.act(state)
        elif hasattr(agent, 'select_action'):
            action, log_prob = agent.select_action(state)
        else:
            raise AttributeError("Agent must have either 'act' or 'select_action' method")
        
        actions.append(action)
        log_probs.append(log_prob)
        
        # Execute step
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward
        
        done = terminated or truncated
        state = next_state
        
        if done:
            break
    
    return rewards, log_probs, states, actions, total_reward


def batch_training_episodes(agent, env, num_episodes, max_steps=1000, device='cpu', print_every=50):
    """
    Executes training for multiple consecutive episodes
    
    Args:
        agent: L'agente da allenare
        env: L'environment
        num_episodes: Numero di episodi
        max_steps: Step massimi per episodio
        device: Device for computations
        print_every: Logging frequency
    
    Returns:
        List of total scores per episode
    """
    episode_scores = []
    scores_deque = deque(maxlen=100)
    
    for episode in range(num_episodes):
        rewards, log_probs, states, actions, total_reward = generic_episode_training(
            agent, env, max_steps, device
        )
        
        episode_scores.append(total_reward)
        scores_deque.append(total_reward)
        
        if episode % print_every == 0:
            avg_score = np.mean(scores_deque) if scores_deque else 0
            print(f'Episode {episode}\tScore: {total_reward:.2f}\tAverage Score: {avg_score:.2f}')
    
    return episode_scores
