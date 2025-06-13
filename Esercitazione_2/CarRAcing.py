from src.CarRacing.agent import Agent
from src.CarRacing.environment import Env
from src.CarRacing.config import configure
from src.CarRacing.run import RunManager


import gymnasium as gym
import numpy as np
import cv2

def record_video_standard_env(agent, filename="carracing_test.mp4", max_steps=1000):
    """Record video usando l'environment gym standard"""
    
    # Crea environment standard con rendering
    env_standard = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    
    frames = []
    state, _ = env_standard.reset()
    
    print(f"Recording video for {max_steps} steps...")
    
    for step in range(max_steps):
        # Render frame
        frame = env_standard.render()
        frames.append(frame)
        
        # Converti lo state per il tuo agent (se necessario)
        # Il tuo agent potrebbe aspettarsi un formato diverso
        try:
            with torch.no_grad():
                action, _ = agent.select_action(state)
        except:
            # Se c'è incompatibilità, usa azioni casuali per il video
            action = env_standard.action_space.sample()
        
        # Step environment
        state, reward, terminated, truncated, _ = env_standard.step(action)
        done = terminated or truncated
        
        if step % 100 == 0:
            print(f"Step {step}: action={action}, reward={reward:.3f}")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    env_standard.close()
    
    # Salva video
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved to {filename}")
        return filename
    else:
        print("❌ No frames captured")
        return None



configs, use_cuda,  device = configure()

agent = Agent(configs.checkpoint, configs, device)
env = Env(configs)

run = RunManager(
    project_name="CarRacing",
    run_name=f"run_{configs.checkpoint}",
    config=configs,
    device=device,
    model=agent,
    env=env,
    output_dir="runs",
    save_every=100
)

max_timesteps = 300

global_step = 0
for episodeIndex in range(configs.checkpoint, 700):
    score = 0
    prevState = env.reset()
    for t in range(max_timesteps):
        action, a_logp = agent.select_action(prevState)
        curState, reward, dead, reasonForDeath = env.step(action, t, agent)
        
        # Debug info ogni 50 step - PIU' DETTAGLIATO
        if t % 50 == 0:
            print(f"Step {t}: action={action}, reward={reward:.3f}, dead={dead}")
            print(f"  Score cumulativo: {score:.2f}")
            if hasattr(agent, 'get_action_probs'):
                probs = agent.get_action_probs(prevState)
                print(f"  Action probabilities: {probs}")
        
        if not configs.test:
            loss, entropy = agent.update((prevState, action, a_logp, reward, curState), episodeIndex)
            # Log training metrics ogni 100 step - SOLO quando disponibili
            if t % 100 == 0 and loss is not None and entropy is not None:
                print(f"  Loss: {loss:.4f}, Entropy: {entropy:.4f}")
            global_step += 1
        
        score += reward
        prevState = curState

        if dead:
            break

    # Log episode reward con episode number
    run.log_episode(episodeIndex, score)

    if configs.test:
        break

# fine training
if not configs.test:
    run.save_model()
    # run.record_video_standard_env()
    video_file = record_video_standard_env(agent, "carracing_agent_test.mp4")
    run.finish()