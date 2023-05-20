import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from spacecraft import Spacecraft
import numpy as np
class DQN(nn.Module):
    # definition of the neural network , 1 hidden layer for the moment
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    # using relu all over it
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
def main():
    # Initialize environment
    env = Spacecraft()

    # Hyperparameters
    EPISODES = 1000
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    GAMMA = 0.99
    LR = 0.03

    # Initialize DQN
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training loop
    for episode in range(EPISODES):
        state, unused = env.reset()
        print(state)
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episode / EPS_DECAY)
        done = False

        while not done:
            # Select action
            if np.random.rand() < eps: # exploration
                action = env.action_space.sample()
            else: # exploitation
                state_tensor = torch.FloatTensor(state)
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            # Take action
            reward = 0.
            next_state, reward, done, truncated, empty = env.step(action)

            # Compute target Q value
            next_state_tensor = torch.FloatTensor(next_state)
            next_q_values = dqn(next_state_tensor)
            target_q_value = reward + GAMMA * torch.max(next_q_values)

            # Compute current Q value
            state_tensor = torch.FloatTensor(state)
            current_q_value = dqn(state_tensor)[action]

            # Compute loss
            loss = criterion(current_q_value, target_q_value)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
        current_dir = os.getcwd()
        time_stamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")
        torch.save(dqn.state_dict(), f"{current_dir}\\model_{time_stamp}.pt")

        Result = {"Energy used": env.en_used, "Propellent used": env.prop_used, "Reward": reward, "Data Transferred": env.data_sent}
        return Result



def load_model(path, env):
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == "__main__":
    Result = main()
    print("Finished")
    print(Result)
