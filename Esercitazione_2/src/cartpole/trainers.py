"""
Classi trainer per REINFORCE e REINFORCE con baseline
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import wandb


class ReinforceTrainer:
    """Trainer per l'algoritmo REINFORCE base"""
    
    def __init__(self, policy, optimizer, env, gamma=0.99, max_t=1000, project_name=None, device='cpu'):
        self.policy = policy
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma
        self.max_t = max_t
        self.project_name = project_name
        self.device = device

        self.scores = []
        self.scores_deque = deque(maxlen=100)
        self.best_score = -float('inf')

        if self.project_name:
            wandb.init(project=self.project_name)

    def compute_returns(self, rewards):
        """Calcola i return scontati e normalizzati"""
        returns = deque(maxlen=self.max_t)
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * (returns[0] if returns else 0.0)
            returns.appendleft(R)
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        return (returns - returns.mean()) / (returns.std() + eps)

    def reinforce(self, num_episodes=500, print_every=50, save_path="best_reinforce.pt"):
        """Esegue il training REINFORCE"""
        for i_episode in range(1, num_episodes + 1):
            saved_log_probs = []
            rewards = []
            # Gymnasium reset restituisce una tupla (observation, info)
            state, _ = self.env.reset()

            for t in range(self.max_t):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)
                # Gymnasium step restituisce (observation, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                done = terminated or truncated
                state = next_state
                if done:
                    break

            total_reward = sum(rewards)
            self.scores.append(total_reward)
            self.scores_deque.append(total_reward)

            returns = self.compute_returns(rewards)

            policy_loss = [-log_prob * R for log_prob, R in zip(saved_log_probs, returns)]
            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            avg_score = np.mean(self.scores_deque)

            if self.project_name:
                wandb.log({
                    'episode': i_episode,
                    'reward': total_reward,
                    'avg_reward': avg_score
                })

            if avg_score > self.best_score:
                self.best_score = avg_score
                torch.save(self.policy.state_dict(), save_path)
                if self.project_name:
                    wandb.run.summary["best_avg_reward"] = avg_score

            if i_episode % print_every == 0:
                print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")

        self.plot_rewards(self.scores)
        return self.scores

    def plot_rewards(self, reward_list, window=100):
        """Plotta i reward del training"""
        plt.figure(figsize=(12, 6))
        plt.plot(reward_list, label='Reward per Episode')

        if len(reward_list) >= window:
            moving_avg = np.convolve(reward_list, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(reward_list)), moving_avg, 
                    label=f'{window}-Episode Moving Avg', linewidth=2)

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()


class ReinforceWithBaselineTrainer(ReinforceTrainer):
    """Trainer per REINFORCE con baseline (value function)"""
    
    def __init__(self, policy, value_network, policy_optimizer, value_optimizer,
                 env, gamma=0.99, max_t=1000, project_name=None, device='cpu'):
        super().__init__(policy, policy_optimizer, env, gamma, max_t, project_name, device)
        self.value_network = value_network
        self.value_optimizer = value_optimizer

    def reinforce(self, num_episodes=500, print_every=50, save_path="best_reinforce_baseline.pt"):
        """Esegue il training REINFORCE con baseline"""
        for i_episode in range(1, num_episodes + 1):
            saved_log_probs = []
            rewards = []
            states = []

            state, _ = self.env.reset()

            for t in range(self.max_t):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)
                states.append(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                done = terminated or truncated
                state = next_state
                if done:
                    break

            returns = self.compute_returns(rewards)
            total_reward = sum(rewards)
            self.scores.append(total_reward)
            self.scores_deque.append(total_reward)

            policy_losses = []
            value_losses = []
            for log_prob, R, state in zip(saved_log_probs, returns, states):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                baseline = self.value_network(state_tensor)
                advantage = R - baseline.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(nn.functional.mse_loss(baseline.squeeze(), 
                                                          torch.tensor(R).to(self.device)))

            # Aggiorna policy
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_losses).sum()
            policy_loss.backward()
            self.optimizer.step()

            # Aggiorna value network
            self.value_optimizer.zero_grad()
            value_loss = torch.stack(value_losses).sum()
            value_loss.backward()
            self.value_optimizer.step()

            avg_score = np.mean(self.scores_deque)

            if self.project_name:
                wandb.log({
                    'episode': i_episode,
                    'reward': total_reward,
                    'avg_reward': avg_score,
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item()
                })

            if avg_score > self.best_score:
                self.best_score = avg_score
                torch.save(self.policy.state_dict(), save_path)
                if self.project_name:
                    wandb.run.summary["best_avg_reward"] = avg_score

            if i_episode % print_every == 0:
                print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")

        self.plot_rewards(self.scores)
        return self.scores
