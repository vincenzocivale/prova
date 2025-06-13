import wandb
import torch
import cv2
import numpy as np
import os
import gymnasium as gym

class RunManager:
    def __init__(self, project_name, run_name, config, device, model, env, output_dir="runs", save_every=100):
        self.device = device
        self.model = model
        self.env = env
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # W&B init
        wandb.init(project=project_name, name=run_name, config=config)
        self.wandb = wandb

        self.video_path = os.path.join(self.output_dir, "test_run.mp4")
        self.save_every = save_every

    def log_training_step(self, episode, step, reward=None, loss=None, entropy=None, **kwargs):
        # kwargs per metriche extra dinamiche (es: lr, grad_norm, ecc)
        metrics = {}
        if reward is not None:
            metrics["reward"] = reward
        if loss is not None:
            metrics["loss"] = loss
        if entropy is not None:
            metrics["entropy"] = entropy
        for k,v in kwargs.items():
            metrics[k] = v
        if metrics:
            self.wandb.log(metrics, step=step or episode)

    def log_episode(self, episode, total_reward):
        self.wandb.log({"episode_reward": total_reward}, step=episode)

    def save_model(self, filename=None, episode=None):
        if filename is None and episode is not None:
            filename = f"model_ep{episode}.pth"
        elif filename is None:
            filename = "model.pth"

        path = os.path.join(self.output_dir, filename)
        torch.save(self.model.net.state_dict(), path)
        self.wandb.save(path)

    def maybe_save_checkpoint(self, episode):
        if episode % self.save_every == 0:
            self.save_model(episode=episode)

    def record_test_video(self, filename="test_video.mp4", max_steps=500):
        """Record a video of the agent playing"""
        try:
            frames = []
            state = self.env.reset()
            
            print(f"Recording test video for {max_steps} steps...")
            
            for step in range(max_steps):
                # Get action from model
                with torch.no_grad():
                    action, _ = self.model.select_action(state)
                
                # Step environment - usa la stessa signature del training
                state, reward, done, reason = self.env.step(action, step, self.model)
                
                # Try to render - remove 'mode' parameter
                try:
                    frame = self.env.render()  # RIMOSSO mode='rgb_array'
                    if frame is not None:
                        frames.append(frame)
                except Exception as render_error:
                    print(f"Warning: Could not render frame at step {step}: {render_error}")
                
                if done:
                    print(f"Episode ended at step {step}: {reason}")
                    break
            
            # Save video if we have frames
            if frames and len(frames) > 0:
                self._save_video(frames, filename)
            else:
                print("No frames captured - creating summary instead")
                print(f"Test episode completed in {step} steps with final score: {reward}")
                
        except Exception as e:
            print(f"Error recording test video: {e}")
            print("Continuing without video recording...")


    def record_video_standard_env(agent, filename="carracing_test.mp4", max_steps=1000):
        """Record video usando l'environment gym standard"""
        
        # Crea environment standard con rendering
        env_standard = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
        
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

    def finish(self):
        self.wandb.finish()