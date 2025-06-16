"""
Modulo training per funzioni di training condivise
"""
import numpy as np
import torch
from collections import deque


def carracing_reinforce_training(agent, env, configs, run_manager, num_episodes=100, print_every=10):
    """
    Funzione di training stile REINFORCE per CarRacing
    Simile alla struttura del training CartPole ma adattata per CarRacing PPO
    
    Args:
        agent: L'agente CarRacing da allenare
        env: L'environment CarRacing
        configs: Configurazioni dell'esperimento
        run_manager: Manager per logging e salvataggio
        num_episodes: Numero di episodi di training
        print_every: Frequenza di print dei risultati
    
    Returns:
        Lista dei punteggi per episodio
    """
    scores = []
    scores_deque = deque(maxlen=100)
    best_score = -float('inf')
    
    print(f"🚗 Inizio training per {num_episodes} episodi...")
    
    for episode in range(configs.checkpoint, configs.checkpoint + num_episodes):
        episode_score = 0
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]  # Storia azioni per environment
        
        # Esegui un episodio completo
        for t in range(configs.max_episode_steps):
            # Seleziona azione tramite policy
            action, log_prob = agent.select_action(state)
            
            # Esegui step nell'environment
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            # Aggiorna il buffer dell'agente (simile a REINFORCE ma con PPO buffer)
            if not configs.test:
                loss, entropy = agent.update((state, action, log_prob, reward, next_state), episode)
            
            episode_score += reward
            state = next_state
            action_history = [action_history[-1], action]
            
            if done:
                break
        
        # Tracking dei risultati
        scores.append(episode_score)
        scores_deque.append(episode_score)
        avg_score = np.mean(scores_deque)
        
        # Log metrics
        run_manager.log_episode(episode, episode_score)
        
        # Salva il miglior modello
        if avg_score > best_score:
            best_score = avg_score
            run_manager.save_model("best_carracing_model.pth")
        
        # Checkpoint periodico
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
    Funzione generica per eseguire un episodio di training
    Utilizzabile per diversi tipi di agenti e environment
    
    Args:
        agent: Agente con metodi select_action e act
        env: Environment gymnasium-compatibile
        max_steps: Numero massimo di step per episodio
        device: Device per computazioni
    
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
        
        # Seleziona azione
        if hasattr(agent, 'act'):
            action, log_prob = agent.act(state)
        elif hasattr(agent, 'select_action'):
            action, log_prob = agent.select_action(state)
        else:
            raise AttributeError("Agent must have either 'act' or 'select_action' method")
        
        actions.append(action)
        log_probs.append(log_prob)
        
        # Esegui step
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
    Esegue training per più episodi consecutivi
    
    Args:
        agent: L'agente da allenare
        env: L'environment
        num_episodes: Numero di episodi
        max_steps: Step massimi per episodio
        device: Device per computazioni
        print_every: Frequenza di logging
    
    Returns:
        Lista dei punteggi totali per episodio
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
