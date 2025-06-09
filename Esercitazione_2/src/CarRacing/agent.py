from .config import Args
from .net import Net
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


class Agent():
    """
    Agent for training
    """

    def __init__(self, episode, args:Args, device):

        transition = np.dtype([('s', np.float64, (args.valueStackSize*args.numberOfLasers + 3*args.actionStack, )), ('a', np.int64), ('a_logp', np.float64),
                    ('r', np.float64), ('s_', np.float64, (args.valueStackSize*args.numberOfLasers + 3*args.actionStack, ))])

        self.args = args
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.buffer_capacity = args.buffer_capacity
        self.prevSaveIndex = episode
        self.batch_size = args.batch_size
        self.training_step = 0
        self.net = Net(args).double().to(device)
        self.device = device
        if episode != 0:
            print("LOADING FROM EPISODE", episode)
            self.net.load_state_dict(torch.load(self.args.saveLocation + 'episode-' + str(episode) + '.pkl'))
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.lastSavedEpisode = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    
    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(state)  # supponendo che la rete restituisca logits (shape [1, 5])
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        a_logp = dist.log_prob(action)
        return action.item(), a_logp.item()

    def save_param(self, episode ):
        self.lastSavedEpisode = episode
        print('-----------------------------------------')
        print("SAVING AT EPISODE", episode)
        print('-----------------------------------------')
        torch.save(self.net.state_dict(), self.args.saveLocation + 'episode-' + str(episode) +  '.pkl')

        

    def update(self, transition, episodeIndex):
        self.buffer[self.counter] = transition
        self.counter += 1
        avg_loss = None
        avg_entropy = None
        
        if self.counter == self.buffer_capacity:
            print("UPDATING WEIGHTS AT EPISODE = ", episodeIndex)
            self.counter = 0
            self.training_step += 1

            s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
            a = torch.tensor(self.buffer['a'], dtype=torch.long).to(self.device)  # discrete action as int64/long
            r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
            s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

            old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

            with torch.no_grad():
                target_v = r + self.args.gamma * self.net(s_)[1]
                advantage = target_v - self.net(s)[1]

            losses = []
            entropies = []

            for _ in range(self.ppo_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                    logits = self.net(s[index])[0]
                    dist = Categorical(logits=logits)
                    a_logp = dist.log_prob(a[index]).view(-1, 1)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * advantage[index]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[index]
                    actorLoss = -torch.min(surr1, surr2).mean()
                    criticLoss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                    loss = actorLoss + 2. * criticLoss

                    entropy = dist.entropy().mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses.append(loss.item())
                    entropies.append(entropy.item())

            avg_loss = sum(losses) / len(losses)
            avg_entropy = sum(entropies) / len(entropies)

            if episodeIndex - self.prevSaveIndex > 10:
                self.save_param(episodeIndex)
                self.prevSaveIndex = episodeIndex

        return avg_loss, avg_entropy
