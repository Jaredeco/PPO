import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym
from collections import deque


class Policy(nn.Module):
    def __init__(self, input_dim, n_actions, pol_type, lr):
        super(Policy, self).__init__()
        self.pol_type = pol_type
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.ac = nn.Linear(256, n_actions)
        self.cri = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.cuda()

    def forward(self, x, softmax_dim=0):
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.pol_type == "actor":
            x = F.softmax(self.ac(x).detach(), dim=softmax_dim)
        elif self.pol_type == "critic":
            x = self.cri(x)
        return x


class PPO:
    def __init__(self, input_dim, n_actions, lr=0.0005):
        self.gamma = 0.98
        self.epsilon = 0.2
        self.actor = Policy(input_dim, n_actions, "actor", lr)
        self.critic = Policy(input_dim, n_actions, "critic", lr)
        self.memory = deque(maxlen=1000)
        self.epochs = 3
        self.lmbda = 0.95

    def store_data(self, data):
        self.memory.append(data)

    def select_action(self, obs):
        probs = self.actor(obs)
        dis = torch.distributions.Categorical(probs)
        action = dis.sample()
        # self.log_prob = dis.log_prob(action)
        return action.item(), probs

    def train(self):
        states, actions, rewards, next_states, probs_act, dones = [i[0] for i in self.memory], [i[1] for i in
                                                                                                self.memory], \
                                                                  [i[2] for i in self.memory], [i[3] for i in
                                                                                                self.memory], \
                                                                  [i[4] for i in self.memory], [i[5] for i in
                                                                                                self.memory],

        states = torch.tensor(states, dtype=torch.float32).cuda()
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).cuda()
        next_states = torch.tensor(next_states, dtype=torch.float32).cuda()
        probs_act = torch.tensor(probs_act, dtype=torch.float32).cuda()
        dones = torch.tensor(dones, dtype=torch.float32).cuda()

        for epoch in range(self.epochs):
            next_critic = self.critic(next_states)
            critic = self.critic(states)
            td_error = rewards + self.gamma * next_critic * (1 - dones).unsqueeze(1)
            delta = td_error - critic
            delta = delta.cpu().detach().numpy()
            advantages = []
            advantage = 0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantages.append(advantage)
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float32).cuda()
            actor_val = self.actor(states, softmax_dim=1)
            actor_val = actor_val.gather(1, actions)
            ratio = torch.exp(torch.log(actor_val) - torch.log(probs_act))
            now_vs_prev = ratio * advantages
            g = torch.clamp(ratio, (1 - self.epsilon), (1 + self.epsilon)) * advantage
            actor_loss = -torch.min(now_vs_prev, g)
            critic_pred = self.critic(states)
            critic_loss = F.smooth_l1_loss(critic_pred, td_error)
            loss = actor_loss + critic_loss
            loss = loss.mean()
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            self.actor.optimizer.step()


num_episodes = 10000
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
agent = PPO(input_dim, n_actions)
T = 20
for episode in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        if episode > 250:
            env.render()
        for t in range(T):
            action, probs = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_data((state, action, reward, next_state, probs[action].item(), done))
            state = next_state
            score += reward
            if done:
                break
        agent.train()
    print(f"Episode {episode}, Score {score}")
