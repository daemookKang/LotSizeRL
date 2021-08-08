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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 1), stride=1)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 1), stride=1)

        self.fc1 = nn.Linear(4 * n, 100)
        #         self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, output_dim)

        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        # torch.nn.init.xavier_uniform_(self.conv3.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)



        self.relu = nn.ReLU()
        self.leay = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = torch.cat([x[0], x[1]], dim=0)
        x2 = torch.cat([x[2], x[3], x[4]], dim=0)

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

        x = x.view(-1, 4 * self.n)

        x = F.relu(self.fc1(x))

        #         x = F.relu(self.fc2(x))

        x = self.fc3(x)

        # print(x)

        return x[0]






