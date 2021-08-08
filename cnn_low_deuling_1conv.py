from datetime import datetime
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import datetime


class NN_FABSched(nn.Module):
    def __init__(self, n, output_dim):
        super(NN_FABSched, self).__init__()
        self.n = n
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5, 1), stride=1)
        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(2, 1), stride=1)
        # self.conv3 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(2, 1), stride=1)

        self.fc1_adv = nn.Linear(5 * n, 18)
        self.fc1_val = nn.Linear(5 * n, 18)
        #         self.fc2 = nn.Linear(500, 100)
        self.fc3_adv = nn.Linear(18, self.output_dim)
        self.fc3_val = nn.Linear(18, 1)

        self.relu = nn.ReLU()
        self.leay = nn.LeakyReLU(0.1)

    def forward(self, x):

        x = x.view(1, 1, -1, self.n)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 5 * self.n)

        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc3_adv(adv)[0]
        val = self.fc3_val(val)[0]
        #         x = F.relu(self.fc2(x))

        val.expand(0,self.output_dim)

        # print(adv , " :: adv")
        # print(val, " ::val")

        q_val = val + adv - adv.mean(0).expand(self.output_dim)

        # print(q_val, "q_val")

        return q_val






