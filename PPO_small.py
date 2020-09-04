import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym


class Policy(nn.Module):
    def __init__(self, input_dim, n_actions, pol_type):
        super(Policy, self).__init__()
        self.pol_type = pol_type
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.ac = nn.Linear(256, n_actions)
        self.cri = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.pol_type == "actor":
            x = self.ac(x)
        elif self.pol_type == "critic":
            x = self.cri(x)
        return x


class PPO:
    def __init__(self, input_dim, n_actions):
        self.gamma = 0.99
        self.epsilon = 0.2
        self.actor = Policy(input_dim, n_actions, "actor")
        self.critic = Policy(input_dim, n_actions, "critic")
        self.log_prob = None
        self.prev_log_prob = None

    def select_action(self, obs):
        probs = F.softmax(self.actor(obs), dim=-1)
        dis = torch.distributions.Categorical(probs)
        action = dis.sample()
        self.log_prob = dis.log_prob(action)
        return action.item()

    def train(self, state, reward, next_state, done):
        if self.prev_log_prob is not None:
            reward = torch.tensor(reward, dtype=torch.float32).cuda()
            critic = self.critic(state)
            next_critic = self.critic(next_state)
            advantage = reward + self.gamma * next_critic * (1 - int(done)) - critic
            now_vs_prev = (self.log_prob - self.prev_log_prob).exp() * advantage
            g = torch.clamp(now_vs_prev, (1 - self.epsilon), (1 + self.epsilon)) * advantage
            actor_loss = -torch.min(now_vs_prev, g)
            critic_loss = torch.mean(advantage ** 2)
            loss = (actor_loss + critic_loss)
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            self.actor.optimizer.step()
        self.prev_log_prob = self.log_prob


num_episodes = 400
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
agent = PPO(input_dim, n_actions)
for episode in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        if episode > 370:
            env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, reward, next_state, done)
        score += reward
    print(episode, score)
