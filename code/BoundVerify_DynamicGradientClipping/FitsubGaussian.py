import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class gaussian_net(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        # input to hidden
        self.i2h = nn.Linear(1, hidden_size)
        # hidden to mean
        self.h2mean = nn.Linear(hidden_size, 1)
        # hidden to std
        self.h2std = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        """forward pass returns the distribution, the mean and variance
        
        """
        x = self.i2h(x)
        mean = self.h2mean(x)
        # std needs to be positive, so use softplus to ensure positive
        std = F.softplus(self.h2std(x))
        # create our normal distribution
        dist = Normal(mean, std)
        
        return dist, mean, std