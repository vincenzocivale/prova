import gymnasium as gym
import cv2
import numpy as np
from CarRacing.config import Args

class CarRacingEnv:
    def __init__(self, configs: Args, render_mode="rgb_array"):
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.configs = configs
        self.env.reset(seed=configs.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.render_enabled = render_mode == "human"

        self.reward_buffer = []
        self.car_position = None  # posizione aggiornata ad ogni reset

    def reset(self):
        self.timestep = 0
        self.reward_buffer.clear()
        img_rgb = self.env.reset()[0]  # gym >=0.26 returns (obs, info)
        
        # Calcolo posizione macchina per il ray casting
        self.car_position = self.find_car_position(img_rgb)

        distances = self._get_laser_distances(img_rgb, self.car_position)

        # Init observation stack
        self.stack = distances * self.configs.valueStackSize
        assert len(self.stack) == self.configs.valueStackSize * self.configs.numberOfLasers
        return np.array(self.stack, dtype=np.float32)

    def step(self, action, last_action=None, second_last_action=None):
        total_reward = 0.0
        done = False
        reason = None
        rgb_state = None

        for _ in range(self.configs.action_repeat):
            rgb_state, reward, terminated, truncated, _ = self.env.step(
                self.configs.actionTransformation(action)
            )
            if self.render_enabled:
                self.env.render()

            # Penalty if off-road
            if self._is_on_green(rgb_state):
                reward -= 0.05

            # Jerk penalty (if info is available)
            if last_action is not None and second_last_action is not None:
                jerk = np.linalg.norm(np.array(last_action) - np.array(second_last_action))
                reward -= 10 * jerk

            # Brake penalty
            reward -= action[2]

            total_reward += reward
            self._store_reward(reward)

            self.timestep += 1

            # Termination conditions
            if self._all_recent_rewards_negative():
                done = True
                reason = "Greenery"
                total_reward -= 10
                break
            elif self.timestep > self.configs.deathThreshold:
                done = True
                reason = "Max timesteps"
                break
            elif terminated or truncated:
                done = True
                reason = "Env terminated"
                break

        distances = self._get_laser_distances(rgb_state, self.car_position)

        # Update stack (drop oldest distances, append new)
        self.stack = self.stack[self.configs.numberOfLasers:] + distances
        assert len(self.stack) == self.configs.valueStackSize * self.configs.numberOfLasers

        return np.array(self.stack, dtype=np.float32), total_reward, done, reason

    def render(self):
        self.env.render()

    def _is_on_green(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        region = gray[66:78, 44:52]
        return region.mean() < 100

    def _get_laser_distances(self, rgb, car_pos):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        cropped = binary[0:83, 0:95]

        x, y = car_pos  # usa posizione attuale auto
        distances = []

        for dx in [-1, -0.5, 0, 0.5, 1]:  # L, ML, M, MR, R
            dist = self._ray_cast(cropped, x, y, dx)
            distances.append(dist)

        return distances

    def _ray_cast(self, img, x, y, dx):
        max_dist = self.configs.maxDistance
        for i in range(1, max_dist):
            xi = int(np.clip(x + dx * i, 0, img.shape[1] - 1))
            yi = int(np.clip(y - i, 0, img.shape[0] - 1))
            if img[yi, xi] == 0:
                return round(np.linalg.norm([xi - x, yi - y]), 2)
        return float(max_dist)

    def _all_recent_rewards_negative(self):
        arr = np.array(self.reward_buffer)
        return arr.size > 0 and np.all(arr < 0)

    def _store_reward(self, reward):
        if len(self.reward_buffer) >= self.configs.deathByGreeneryThreshold:
            self.reward_buffer.pop(0)
        self.reward_buffer.append(reward)

    def find_car_position(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
        else:
            # fallback
            return 48, 73
