import math
import random
from collections import deque

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class NoisyLinear(nn.Module):
    # independent gaussian case
    def __init__(self, in_features, out_features, std_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.std_init)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(self.std_init)

    def forward(self, x):
        self.weight_epsilon.data.normal_()
        self.bias_epsilon.data.normal_()
        return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class ReplayBuffer(object):
    """
    后面尽量使用ExperimentReplayBuffer，这个类暴露了experiment的细节，
    而很多dqn方法使用的experiment不进相同
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state[np.newaxis, :], action, reward, next_state[np.newaxis, :], done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.vstack(state), action, reward, np.vstack(next_state), done

    def __len__(self):
        return len(self.buffer)


class ExperimentReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, experiment):
        self.buffer.append(experiment)

    def sample(self, batch_size):
        # 把解开操作放在外面，由不同的dqn方法自己处理
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state[np.newaxis, :], action, reward, next_state[np.newaxis, :], done))
        else:
            self.buffer[self.pos] = (state[np.newaxis, :], action, reward, next_state[np.newaxis, :], done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in indices])
        samples = np.vstack(state), action, reward, np.vstack(next_state), done

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def optimal_q_and_action(self, state):
        out = self.forward(state)
        return out.max(1)[0], out.max(1)[1]


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return val + adv - adv.mean()

    def optimal_q_and_action(self, state):
        out = self.forward(state)
        return out.max(1)[0], out.max(1)[1]


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, self.n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def optimal_q_and_action(self, state):
        out = self.forward(state)
        return out.max(1)[0], out.max(1)[1]


class BootstrappedDQN(nn.Module):
    def __init__(self, input_shape, n_actions, k):
        super(BootstrappedDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.bootstrap_heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            ) for _ in range(k)]
        )
        self.to(DEVICE)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        out = []
        x = self.conv(x)
        x = x.view(1, x.size(0), -1)
        for h in self.bootstrap_heads:
            out.append(h(x))
        return torch.cat(out)  # K*B*A

    def optimal_q_and_action(self, state):
        out = self.forward(state)
        return out.max(-1)[0], out.max(-1)[1]  # K*B
