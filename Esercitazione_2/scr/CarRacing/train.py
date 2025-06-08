import gym
import torch
import numpy as np
import wandb
import cv2
from agent import Agent  # la tua classe Agent rifattorizzata
from config import Args, configure

def train_agent(num_episodes=1000):
    # Configurazione e device
    args = configure()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inizializza wandb
    wandb.init(project="car-racing-ppo", config=vars(args))
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array")  # render_mode serve per catturare frame
    agent = Agent(episode=0, args=args, device=device)

    for episode in range(num_episodes):
        state = env.reset()[0]  # Gym v26+ reset restituisce tuple (obs, info)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Lo stato va preprocessato se serve per la rete (dipende dal tuo input)
            action, a_logp = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Costruisci la transizione e aggiorna agent
            transition = (state, action, a_logp, reward, next_state)
            agent.update(transition, episode)

            state = next_state
            total_reward += reward
            steps += 1
        
        # Log su W&B
        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "training_step": agent.training_step
        })

        print(f"Episode {episode}: reward = {total_reward:.2f}, steps = {steps}")

        # Salva modello ogni 50 episodi
        if episode % 50 == 0:
            agent.save_param(episode)
    
    env.close()
    wandb.finish()
    print("Training completo!")

def test_agent(video_path="car_racing_test.mp4", episode=0):
    args = configure()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    agent = Agent(episode=episode, args=args, device=device)  # carica modello allenato

    frames = []
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        frames.append(frame)

        state = next_state
        total_reward += reward
    
    env.close()

    # Salva video con OpenCV
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    for f in frames:
        # OpenCV usa BGR, Gym restituisce RGB, convertiamo
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    print(f"Test finito! Reward totale: {total_reward:.2f}")
    print(f"Video salvato in: {video_path}")


