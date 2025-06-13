from .agent import Agent
import gymnasium as gym
import cv2
import numpy as np
from .config import Args

class Env():
    def __init__(self, configs: Args):
        self.env = gym.make('CarRacing-v3', continuous=False)
        self.configs = configs
        
        self.env.reset(seed=configs.seed)
        self.env.action_space.seed(configs.seed)
        self.env.observation_space.seed(configs.seed)

        self.previousRewards = []
        self.rewards = []
        self.rewardThresh = self.env.spec.reward_threshold

        # Buffer per distanze e azioni
        self.laser_history = []
        self.action_history = []

        # Step globale per jerk penalty
        self.global_step = 0

        # Controllo rendering
        self.render_enabled = False

    def reset(self):
        self.die = False
        self.rewards = []
        self.global_step = 0
        
        obs = self.env.reset()[0]
        distances, _ = self.preprocess(obs)

        self.laser_history = [distances for _ in range(self.configs.valueStackSize)]
        self.action_history = [0 for _ in range(self.configs.actionStack)]

        return self.build_stack()

    def step(self, action, last_action, second_last_action):
        finalReward = 0
        death = False
        reason = 'NULL'
        rgbState = None
        green_counter = 0

        self.action_history = self.action_history[1:] + [action]

        for i in range(self.configs.action_repeat):
            rgbState, reward, terminated, truncated, _ = self.env.step(action)
            self.global_step += 1

            if self.render_enabled:
                self.env.render()

            done = terminated or truncated

            # Penalità leggera per zone verdi
            if self.checkGreen(rgbState):
                green_counter += 1
                reward -= 0.03
            else:
                green_counter = 0

            # Penalità sul jerk 
            jerkPenalty = 0.15 * abs(action - last_action) if self.global_step > 1000 else 0

            progress_bonus = 0.1 if self.configs.actionTransformation(action)[0] > 0 else 0.0

            reward -= jerkPenalty
            reward += progress_bonus

            # Accumula reward (non moltiplichiamo arbitrariamente)
            finalReward += reward

            self.storeRewards(reward)

            if done:
                death = True
                reason = 'Environment termination'
                break

            if (self.configs.total_episodes > 100 and
                self.checkExtendedPenalty() and i > 50):
                death = True
                reason = 'Greenery'
                finalReward -= 2 
                break

            if green_counter > 30:
                death = True
                reason = 'Extended Greenery'
                finalReward -= 2  
                break

        # Media finale del reward (opzionale ma consigliata)
        finalReward /= self.configs.action_repeat

        distances, _ = self.preprocess(rgbState)
        self.laser_history = self.laser_history[1:] + [distances]

        return self.build_stack(), finalReward, death, reason


    def build_stack(self):
        distance_stack = []
        for d in self.laser_history:
            distance_stack.extend(d)

        action_stack = []
        for a in self.action_history:
            encoded = self.configs.actionTransformation(a)
            action_stack.extend(encoded)

        full_stack = distance_stack + action_stack

        expected = self.configs.valueStackSize * self.configs.numberOfLasers + 3 * self.configs.actionStack
        if len(full_stack) != expected:
            print(f"WARNING: Stack size mismatch! Got {len(full_stack)}, expected {expected}")
            if len(full_stack) < expected:
                full_stack.extend([0.0] * (expected - len(full_stack)))
            else:
                full_stack = full_stack[:expected]

        return np.array(full_stack, dtype=np.float32)

    def checkGreen(self, img_rgb):
        _, gray = self.preprocess(img_rgb)
        region = gray[66:78, 44:52]
        return region.mean() < 100

    def render(self, *args):
        self.env.render(*args)

    def checkPixelGreen(self, pixel):
        return pixel < 10  # Soglia più robusta

    def preprocess(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        gray = gray[0:83, 0:95]
        temp = gray.copy()
        temprgb = rgb.copy()[0:83, 0:95]

        x, y = 48, 73
        locs = [None] * 5

        for i in range(95):
            if None not in locs:
                break

            checks = [
                (0, (min(max(0, y - i), 82), min(max(x - i, 0), 94))),
                (4, (min(max(0, y - i), 82), max(min(x + i, 94), 0))),
                (2, (min(max(0, y - i), 82), x)),
                (3, (min(max(0, y - i), 82), max(min(x + i // 2, 94), 0))),
                (1, (min(max(0, y - i), 82), min(max(x - i // 2, 0), 94)))
            ]

            for idx, chk in checks:
                if locs[idx] is None and self.checkPixelGreen(temp[chk]):
                    locs[idx] = chk

        distances = []
        for pt in locs:
            if pt is None:
                pt = (self.configs.maxDistance, self.configs.maxDistance)
            dist = round(np.linalg.norm(np.array(pt) - np.array((y, x))), 2)
            distances.append(dist if dist != 0 else self.configs.maxDistance)

        return distances, gray

    def checkExtendedPenalty(self):
        r = np.array(self.rewards)
        return len(r) > 0 and np.all(r < 0)

    def storeRewards(self, reward):
        if len(self.rewards) >= self.configs.deathByGreeneryThreshold:
            self.rewards.pop(0)
        self.rewards.append(reward)
