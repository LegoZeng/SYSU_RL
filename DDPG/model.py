import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()
        # self.bn0 = nn.BatchNorm1d(state_dim)
        self.l1 = nn.Linear(state_dim, 400)
        # self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        # self.bn1 = nn.BatchNorm1d(400+action_dim)
        self.l2 = nn.Linear(400 + action_dim, 300)
        # self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        # self.bn2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 1)
        # self.l3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):

        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()

        self.action_lim = action_lim

        # self.bn0 = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim, 400)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        # self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, action_dim)
        # self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        action = action * self.action_lim
        return action
