import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_policy(observation_space, action_space, **kwargs):
    class CustomPolicy(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], act_dim)

        def forward(self, obs):
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    policy = CustomPolicy(observation_space.shape[0], action_space.n)
    return policy
