import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from spacecraft import Spacecraft
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_space + action_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main(actor=None, critic=None):
    # Initialize environment
    env = Spacecraft()

    # Hyperparameters
    EPISODES = 2500
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 0.01
    GAMMA = 0.99
    LR = 0.03
    TAU = 0.01

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if actor and critic:
        actor = actor.to(device)
        critic = critic.to(device)
    else:
        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    target_actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    target_critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)
    criterion = nn.MSELoss()

    writer = SummaryWriter(
        f'runs/spacecraft_experiment_1_{str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")}')

    iteration = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        done = False

        while not done:
            actor.eval()
            with torch.no_grad():
                action = actor(state)
            actor.train()

            noise = torch.normal(mean=0., std=0.2, size=action.shape).to(device)
            action = (action + noise).clamp(env.action_space.low, env.action_space.high)

            next_state, reward, done, _, _ = env.step(action.cpu().numpy())
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor([reward]).to(device)

            target_q = target_critic(next_state, target_actor(next_state))
            target_value = reward + GAMMA * target_q
            expected_value = critic(state, action)

            critic_loss = criterion(expected_value, target_value.detach())

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            actor_loss = -critic(state, actor(state)).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

            state = next_state
            iteration += 1

if __name__ == "__main__":
    main()

