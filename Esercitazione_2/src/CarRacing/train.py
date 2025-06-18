import wandb
import numpy as np
from time import sleep
from CarRacing.agent import Agent
from CarRacing.environment import CarRacingEnv
from CarRacing.config import configure

def setup_wandb(configs):
    wandb.init(
        project=configs.wandb_project,
        name=f"run-ep{configs.checkpoint}",
        config=configs.getParamsDict(),
        mode="disabled" if configs.test else "online"
    )

def training_loop(agent, env, configs):
    for episode in range(configs.checkpoint, 100000):
        score = 0
        state = env.reset()
        
        # Initialize action history for training loop
        action_history = [np.zeros(3), np.zeros(3)]  # [second_last, last]
        
        for t in range(configs.max_episode_steps):
            action, logp = agent.select_action(state)
            
            # OPTION 1: Use the step() method that accepts previous actions
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            # OPTION 2: Alternative - use step_with_steps if you prefer internal handling
            # next_state, reward, done, death_reason = env.step_with_steps(action, t)
            
            if not configs.test:
                agent.update((state, action, logp, reward, next_state), episode)
            
            score += reward
            state = next_state
            
            # Update action history
            action_history = [action_history[-1], action]  # [last, current]
            
            if done:
                print(f"\n[DEAD] Ep={episode}, Score={score:.2f}, Steps={t}, Reason: {death_reason}")
                break
        
        # Log metrics
        if not configs.test:
            wandb.log({
                "score": score,
                "episode_length": t,
                "death_reason": death_reason
            }, step=episode)
        
        print(f"Episode {episode} | Score: {score:.2f} | Steps: {t}")
        print("-" * 40)

# ALTERNATIVE VERSION WITH DEBUGGING
def training_loop_debug(agent, env, configs):
    for episode in range(configs.checkpoint, 100000):
        score = 0
        state = env.reset()
        action_history = [np.zeros(3), np.zeros(3)]
        
        # Debug: verify state dimensions
        print(f"Initial state shape: {state.shape}")
        expected_size = configs.valueStackSize * configs.numberOfLasers + 3 * configs.actionStack
        print(f"Expected size: {expected_size}, Actual size: {len(state)}")
        
        reward_breakdown = {"base": 0, "green": 0, "jerk": 0, "brake": 0}
        
        for t in range(configs.max_episode_steps):
            action, logp = agent.select_action(state)
            
            # Debug: log action
            if t < 5:  # Log first 5 steps
                print(f"Step {t}: Action = {action}")
            
            next_state, reward, done, death_reason = env.step(
                action, action_history[-1], action_history[-2]
            )
            
            # Debug: breakdown reward (if you implement detailed logging in env)
            if hasattr(env, 'last_reward_breakdown'):
                for key, val in env.last_reward_breakdown.items():
                    reward_breakdown[key] += val
            
            if not configs.test:
                agent.update((state, action, logp, reward, next_state), episode)
            
            score += reward
            state = next_state
            action_history = [action_history[-1], action]
            
            if done:
                print(f"\n[DEAD] Ep={episode}, Score={score:.2f}, Steps={t}, Reason: {death_reason}")
                print(f"Reward breakdown: {reward_breakdown}")
                break
        
        if not configs.test:
            log_dict = {
                "score": score,
                "episode_length": t,
                "death_reason": death_reason
            }
            log_dict.update({f"reward_{k}": v for k, v in reward_breakdown.items()})
            wandb.log(log_dict, step=episode)
        
        print(f"Episode {episode} | Score: {score:.2f} | Steps: {t}")
        print("-" * 40)
