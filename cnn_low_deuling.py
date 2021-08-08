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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(2, 1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(2, 1), stride=1)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(2, 1), stride=1)

        self.fc1_adv = nn.Linear(10 * n, 50)
        self.fc1_val = nn.Linear(10 * n, 50)
        #         self.fc2 = nn.Linear(500, 100)
        self.fc3_adv = nn.Linear(50, self.output_dim)
        self.fc3_val = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.leay = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = torch.cat([x[0], x[1]], dim=0)
        x2 = torch.cat([x[3], x[4]], dim=0)

        x1 = x1.view(1, 1, -1, self.n)
        x2 = x2.view(1, 1, -1, self.n)
        # print(torch.randn(1, 1, 4, 2))
        #         print(x)
        x1 = F.relu(self.conv1(x1))
        #         print(x)
        #         print("!!!!!!!!!")
        x2 = F.relu(self.conv2(x2))
        #         print(x1)
        #         print(x2)

        x = torch.cat([x1, x2], dim=2)
        #         print(x)
        #         print("!!!!!")
        x = F.relu(self.conv3(x))

        x = x.view(-1, 10 * self.n)

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






