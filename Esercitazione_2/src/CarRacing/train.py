import wandb
import numpy as np
from time import sleep

from CarRacing.agent import Agent
from CarRacing.environment import CarRacingEnv
from CarRacing.config import configure


def setup_wandb(configs):
    wandb.init(
        project=configs.wandb_project,  # aggiungi questa variabile al config
        name=f"run-ep{configs.checkpoint}",
        config=configs.getParamsDict(),
        mode="disabled" if configs.test else "online"
    )


def training_loop(agent, env, configs):
    for episode in range(configs.checkpoint, 100000):
        score = 0
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]  # second_last, last actions

        for t in range(configs.max_episode_steps):
            action, logp = agent.select_action(state)
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )

            if not configs.test:
                agent.update((state, action, logp, reward, next_state), episode)

            score += reward
            state = next_state
            action_history = [action_history[-1], action]  # keep last 2

            if done:
                print(f"\n[DEAD] Ep={episode}, Score={score:.2f}, Steps={t}, Reason: {death_reason}")
                break

        wandb.log({"score": score}, step=episode)
        print(f"Episode {episode} | Score: {score:.2f}\n" + "-" * 40)