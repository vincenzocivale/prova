import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from config import Args
from net import Net


class Agent:
    def __init__(self, episode: int, args: Args, device):
        self.args = args
        self.device = device
        self.episode = episode

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size

        self.training_step = 0
        self.lastSavedEpisode = episode
        self.prevSaveIndex = episode

        self.net = Net(args).double().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.buffer = self._init_buffer()
        self.counter = 0

        if episode != 0:
            self._load_model(episode)

    def _init_buffer(self):
        transition = np.dtype([
            ('s', np.float64, (self.args.valueStackSize * self.args.numberOfLasers + 3 * self.args.actionStack,)),
            ('a', np.float64, (3,)),
            ('a_logp', np.float64),
            ('r', np.float64),
            ('s_', np.float64, (self.args.valueStackSize * self.args.numberOfLasers + 3 * self.args.actionStack,))
        ])
        return np.empty(self.buffer_capacity, dtype=transition)

    def _load_model(self, episode):
        print("LOADING FROM EPISODE", episode)
        model_path = f"{self.args.saveLocation}episode-{episode}.pkl"
        self.net.load_state_dict(torch.load(model_path))

    def _save_model(self, episode):
        print(f"\nSAVING AT EPISODE {episode}\n" + '-' * 40)
        model_path = f"{self.args.saveLocation}episode-{episode}.pkl"
        torch.save(self.net.state_dict(), model_path)
        self.lastSavedEpisode = episode

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state_tensor)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.squeeze().cpu().numpy(), log_prob.item()

    def update(self, transition, episodeIndex):
        self.buffer[self.counter] = transition
        self.counter += 1

        if self.counter < self.buffer_capacity:
            return

        print("UPDATING WEIGHTS AT EPISODE =", episodeIndex)
        self.counter = 0
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.args.gamma * self.net(s_)[1]
            advantage = target_v - self.net(s)[1]

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                self._update_step(s[index], a[index], old_a_logp[index], advantage[index], target_v[index])

        if episodeIndex - self.prevSaveIndex > 10:
            self._save_model(episodeIndex)
            self.prevSaveIndex = episodeIndex

    def _update_step(self, s_batch, a_batch, old_logp_batch, adv_batch, target_v_batch):
        (alpha, beta), value = self.net(s_batch)
        dist = Beta(alpha, beta)
        logp = dist.log_prob(a_batch).sum(dim=1, keepdim=True)
        ratio = torch.exp(logp - old_logp_batch)

        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_batch
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.smooth_l1_loss(value, target_v_batch)

        loss = actor_loss + 2. * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
