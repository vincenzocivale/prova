import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from config import Args
from envirorment import CarRacingEnv
from agent import Agent


class Trainer:
    def __init__(self, env, agent, args: Args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize environment and agent
        self.env = env
        self.agent = agent
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = deque(maxlen=100)
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Target episodes: {self.args.max_episodes}")
        
        for episode in range(self.args.start_episode, self.args.max_episodes):
            episode_reward, episode_length, termination_reason = self._run_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.recent_rewards.append(episode_reward)
            
            # Log progress
            if episode % self.args.log_interval == 0:
                self._log_progress(episode, episode_reward, episode_length, termination_reason)
            
            # Plot progress
            if episode % self.args.plot_interval == 0 and episode > 0:
                self._plot_progress(episode)
                
            # Early stopping check
            if self._should_stop_early():
                print(f"Early stopping at episode {episode}")
                break
                
        print("Training completed!")
        self._save_final_results()
        
    def _run_episode(self, episode):
        """Run a single episode"""
        state = self.env.reset()
        
        # Initialize action history for stacking
        action_history = [np.zeros(3) for _ in range(self.args.actionStack)]
        
        episode_reward = 0
        episode_length = 0
        done = False
        termination_reason = None
        
        while not done:
            # Create state with action history
            full_state = self._create_full_state(state, action_history)
            
            # Select action
            action, log_prob = self.agent.select_action(full_state)
            
            # Get previous actions for penalties
            last_action = action_history[-1] if len(action_history) > 0 else None
            second_last_action = action_history[-2] if len(action_history) > 1 else None
            
            # Take step in environment
            next_state, reward, done, termination_reason = self.env.step(
                action, last_action, second_last_action
            )
            
            # Update action history
            action_history.append(action)
            if len(action_history) > self.args.actionStack:
                action_history.pop(0)
            
            # Create next full state
            next_full_state = self._create_full_state(next_state, action_history)
            
            # Store transition
            transition = (full_state, action, log_prob, reward, next_full_state)
            self.agent.update(transition, episode)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            episode_length += 1
            
        return episode_reward, episode_length, termination_reason
    
    def _create_full_state(self, laser_state, action_history):
        """Combine laser distances with action history"""
        # Flatten action history
        action_flat = np.concatenate(action_history)
        
        # Combine laser state with action history
        full_state = np.concatenate([laser_state, action_flat])
        
        return full_state
    
    def _log_progress(self, episode, reward, length, reason):
        """Log training progress"""
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        max_reward = max(self.episode_rewards) if self.episode_rewards else 0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:7.2f} | "
              f"Length: {length:3d} | "
              f"Avg100: {avg_reward:7.2f} | "
              f"Max: {max_reward:7.2f} | "
              f"Reason: {reason}")
    
    def _plot_progress(self, episode):
        """Plot training progress"""
        if len(self.episode_rewards) < 10:
            return
            
        plt.figure(figsize=(12, 4))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.plot(self._smooth_curve(self.episode_rewards, window=50), 'r-', linewidth=2)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.plot(self._smooth_curve(self.episode_lengths, window=50), 'r-', linewidth=2)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.args.saveLocation}progress_episode_{episode}.png')
        plt.close()
    
    def _smooth_curve(self, data, window=50):
        """Smooth curve for plotting"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2)
            smoothed.append(np.mean(data[start:end]))
        return smoothed
    
    def _should_stop_early(self):
        """Check if training should stop early"""
        if len(self.recent_rewards) < 100:
            return False
            
        # Stop if average reward over last 100 episodes exceeds threshold
        avg_reward = np.mean(list(self.recent_rewards))
        return avg_reward >= self.env.reward_threshold
    
    def _save_final_results(self):
        """Save final training results"""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'final_avg_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        }
        
        np.save(f'{self.args.saveLocation}training_results.npy', results)
        print(f"Results saved to {self.args.saveLocation}training_results.npy")

