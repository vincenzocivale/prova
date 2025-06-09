import wandb
import torch
import cv2
import numpy as np
import os

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

    def record_test_video(self, filename="test_run.mp4", max_steps=1000):
        state = self.env.reset()
        frames = []

        for _ in range(max_steps):
            with torch.no_grad():
                action, _ = self.model.select_action(state)
            state, reward, done, _ = self.env.step(action)

            frame = self.env.render(mode='rgb_array')
            frames.append(frame)

            if done:
                break

        if frames:
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(self.video_path, fourcc, 30, (width, height))
            for frame in frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()
            self.wandb.save(self.video_path)

    def finish(self):
        self.wandb.finish()