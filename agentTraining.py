import gymnasium
from PolicyNet import PolicyNet
from torch.distributions.categorical import Categorical
from trainingConstants import MAX_TRAJECTORIES, MAX_STEPS, GAMMA, HIDDEN_SIZE, LEARNING_RATE
import torch
import numpy as np


def compute_return(trajectory):
    g_vals = []
    for i in range(len(trajectory)):
        g_val = 0
        power = 0
        for j in range(i, len(trajectory)):
            _, reward, _ = trajectory[j]
            g_val += GAMMA ** power * reward
            power += 1
        g_vals.append(g_val)

    return g_vals


def initialize_env():
    env = gymnasium.make('CartPole-v1', render_mode='human')
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    policy_net = PolicyNet(observation_space, action_space, hidden_size=HIDDEN_SIZE)

    return env, policy_net


def collect_trajectory(env, policy_net):
    current_state = env.reset()
    current_state = current_state[0]
    transitions = []
    for i in range(MAX_STEPS):
        env.render()
        action_prob = policy_net(torch.from_numpy(current_state).float())
        distribution = Categorical(action_prob)
        action = distribution.sample()
        previous_state = current_state

        current_state, reward, done, _, _ = env.step(action.item())
        transitions.append((previous_state, action, i + 1))

        if done:
            break
            env.close()

    return transitions, len(transitions)


def improve_policy(g_vals, policy_net, trajectory, optimizer):
    state_batch = torch.Tensor([s for (s, a, r) in trajectory])
    action_batch = torch.Tensor([a for (s, a, r) in trajectory])
    g_vals = torch.Tensor(g_vals)

    pred_batch = policy_net(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

    loss = -torch.sum(torch.log(prob_batch) * g_vals)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


env, policy_net = initialize_env()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
scores = []

for i in range(MAX_STEPS):
    trajectory, score = collect_trajectory(env, policy_net)
    scores.append(score)
    g_vals = compute_return(trajectory)
    print(score)
    improve_policy(g_vals, policy_net, trajectory, optimizer)
    if i % 50 == 0 and i > 0:
        print(f"Average Score after {i} trajectories: {np.mean(scores[-50:-1])}")

print(np.mean(scores[-50:-1]))

#torch.save(policy_net.state_dict(), "policy_net.pth")
