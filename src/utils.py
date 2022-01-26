from collections import namedtuple
import random
import numpy as np
from math import sqrt
import torch.nn as nn


Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayBuffer(object):
    """
    A class to represent a Replay Buffer - used to store history of our actions 

    Experience - a tuple of values for each: ('state', 'action', 'next_state', 'reward', 'terminal') # Could reimplment in pandas
    ...

    Attributes
    ----------
    buffer: 2D list aka. list(Experience)
        filled with Nones by default
        
    size: int
        Size of the Buffer
        
    position : int
        position of the Buffer (index)

    Methods
    -------
    push(Experience):
        append the current buffer by a tuple with current state the neural network is in called Experience: ('state', 'action', 'next_state', 'reward', 'terminal')
    """


    def __init__(self, size):
        self.buffer = []
        self.size = size
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.size

    def __len__(self):
        return len(self.buffer)

class OU_Noise:
    '''
    
    '''
    def __init__(self, mu, sigma, theta=.15, dt=1e-2,):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def get_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def fan_in_uniform_init(tensor):
    """Utility function for initializing actor and critic"""
    w = 1. / np.sqrt(tensor.size(-1))
    nn.init.uniform_(tensor, -w, w)