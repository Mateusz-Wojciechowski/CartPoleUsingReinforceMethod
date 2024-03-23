import torch
import gymnasium
from PolicyNet import PolicyNet
from trainingConstants import HIDDEN_SIZE
from torch.distributions.categorical import Categorical


def choose_action(policy_net, state):
    state = torch.from_numpy(state).float()
    action_probs = policy_net(state)
    distribution = Categorical(action_probs)
    action = distribution.sample()
    return action


def run_with_policy(env, policy_net):
    state = env.reset()
    state = state[0]
    done = False
    steps = 0

    while not done:
        env.render()
        action = choose_action(policy_net, state)
        current_state, _, done, _, _ = env.step(action.item())
        state = current_state
        steps += 1

    env.close()
    return steps


if __name__ == '__main__':
    env = gymnasium.make('CartPole-v1', render_mode="human")
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    policy_net = PolicyNet(observation_space, action_space, hidden_size=HIDDEN_SIZE)

    policy_net.load_state_dict(torch.load('policy_net.pth'))

    steps = run_with_policy(env, policy_net)
    print(steps)
