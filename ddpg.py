import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from collections import deque
import random
import gym


# noise
# initialize a random process N for action exploration
class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# critic network for evaluating actions
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


# actor network for performing actions
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        return x


class Agent(object):
    def __init__(self, env):
        self.gamma = 0.99
        self.tau = 1
        self.max_size = 1000000
        self.memory = deque(maxlen=self.max_size)
        self.batch_size = 64
        self.n_actions = env.action_space.shape[0]
        self.input_dim = env.observation_space.shape[0]
        # µ(s|θµ)
        self.actor = ActorNetwork(self.input_dim, self.n_actions)
        # Q(s, a|θQ)
        self.critic = CriticNetwork(self.input_dim, self.n_actions)
        # target networks
        # µ'(s|θµ')
        self.target_actor = ActorNetwork(self.input_dim, self.n_actions)
        # Q'(s, a|θQ')
        self.target_critic = CriticNetwork(self.input_dim, self.n_actions)
        # initialize a random process N for action exploration
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        self.update_network_parameters()

    def choose_action(self, observation):
        # select action at = µ(st|θµ) + Nt according to the current policy and exploration noise
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).cuda()
        mu = self.actor.forward(observation).cuda()
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).cuda()
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        # store transition (st, at, rt, st+1) in R
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # sample a random minibatch of N transitions (si, ai, ri, si+1) from R
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], \
                                                  [i[3] for i in batch], [i[4] for i in batch],

        reward = torch.tensor(reward, dtype=torch.float).cuda()
        done = torch.tensor(done).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        state = torch.tensor(state, dtype=torch.float).cuda()
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(next_state)
        critic_value_ = self.target_critic.forward(next_state, target_actions)
        critic_value = self.critic.forward(state, action)
        # yi = ri + γQ'(si+1, µ'(si+1|θµ')|θQ')
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = torch.tensor(target).cuda()
        target = target.view(self.batch_size, 1)
        # update critic by minimizing the loss: L = 1/N Σi(yi − Q(si, ai|θQ))2
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()
        # update the actor policy using the sampled policy gradient
        # ∇θµ J ≈ 1/N Xi ∇aQ(s, a|θQ)|s=si,a=µ(si)∇θµ µ(s|θµ)|si
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self):
        actor = dict(self.actor.named_parameters())
        critic = dict(self.critic.named_parameters())
        target_actor = dict(self.target_actor.named_parameters())
        target_critic = dict(self.target_critic.named_parameters())
        # θQ' ← τθQ + (1 − τ )θQ'
        for param in critic:
            critic[param] = self.tau * critic[param].clone() + (1 - self.tau) * target_critic[param].clone()
        # θµ' ← τθµ + (1 − τ )θµ'
        for param in actor:
            actor[param] = self.tau * actor[param].clone() + (1 - self.tau) * target_actor[param].clone()
        self.target_critic.load_state_dict(critic)
        self.target_actor.load_state_dict(actor)


env = gym.make('MountainCarContinuous-v0')
agent = Agent(env=env)
np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward
        obs = next_state
    score_history.append(score)

    print(f'Episode , {i}, Score {score}, Last 100 games avg {np.mean(score_history[-100:])}')

