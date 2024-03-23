from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obsv_space, action_space, hidden_size):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obsv_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        output = self.policy_net(x)
        return output
