import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    """Class for neural network
    Attributes:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
    """

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x