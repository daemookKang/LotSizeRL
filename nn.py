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
    def __init__(self, input_dim,output_dim):
        super(NN_FABSched, self).__init__()

        self.theta1 = nn.Linear(input_dim, 64)
        self.theta2 = nn.Linear(64, 32)
        self.theta3 = nn.Linear(32, 16)
        self.theta4 = nn.Linear(16, 8)
        self.theta5 = nn.Linear(8, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.theta1(x))
        x = F.relu(self.theta2(x))
        x = F.relu(self.theta3(x))
        x = F.relu(self.theta4(x))
        x = self.theta5(x)

        return x






