import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from utils import fan_in_uniform_init

WEIGHTS_FINAL = 3e-3

# Define at the beginning so triggers upon import
if torch.cuda.is_available() : 
    device = "cuda"
else : 
    deivce = "cpu"


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.ln1 = nn.Linear(num_inputs, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        # In the second layer the actions are inserted as well
        self.ln2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output layer (single value)
        self.ln3 = nn.Linear(hidden_size[1], 1)

        # weight Init
        fan_in_uniform_init(self.ln1.weight)
        fan_in_uniform_init(self.ln2.weight)
        nn.init.uniform_(self.ln3.weight, -WEIGHTS_FINAL, WEIGHTS_FINAL)

    def forward(self, x, actions):
        # Layer 1
        x = self.ln1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), dim=1)
        x = self.ln2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        Q_Value = self.ln3(x)
        return Q_Value


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.norm0 = nn.BatchNorm1d(num_inputs)
        
        # Layer 1
        self.ln1 = nn.Linear(num_inputs, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        self.ln2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output Layer
        self.l3 = nn.Linear(hidden_size[1], num_outputs)

        # weights Init
        fan_in_uniform_init(self.ln1.weight)
        fan_in_uniform_init(self.ln2.weight)
        nn.init.uniform_(self.l3.weight, -WEIGHTS_FINAL, WEIGHTS_FINAL)

    def forward(self, x):
        x = self.norm0(x)
        
        # Layer 1
        x = self.ln1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.ln2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        return torch.tanh(self.l3(x))