import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from spacecraft import Spacecraft
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# less learning rate, much more episode
class DQN(nn.Module):
    # definition of the neural network , 1 hidden layer for the moment
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)
        #self.fc4 = nn.Linear(128, 128)
        #self.fc5 = nn.Linear(128, 128)
        #self.fc6 = nn.Linear(128, 128)
        #self.fc7 = nn.Linear(128, action_space)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    # using relu all over it
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))
        #x = torch.relu(self.fc5(x))
        #x = torch.relu(self.fc6(x))
        return self.fc3(x)


def main(model=None):
    # Initialize environment
    env = Spacecraft()

    # Hyperparameters
    EPISODES = 1000
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 0.01
    GAMMA = 0.99
    LR = 0.001

    # Initialize DQN
    if model:
        dqn = model
    else:
        dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    criterion = nn.MSELoss()

    writer = SummaryWriter(
        f'runs/spacecraft_experiment_1_{str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")}')

    # writer.add_graph(dqn)
    iteration = 0

    # Training loop
    for episode in range(EPISODES):
        state, unused = env.reset()
        print(f"Episode {episode} from {EPISODES}")
        eps = 0.9 - (episode*0.001)
        done = False
        truncated = False

        writer.add_scalar('eps', eps, episode)

        while not done and not truncated:
            # Select action
            if np.random.rand() < eps: # exploration
                action = env.action_space.sample()
            else: # exploitation
                state_tensor = torch.FloatTensor(state)
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            # Take action
            reward = 0.
            next_state, reward, done, truncated = env.step(action)

            # Compute target Q value
            next_state_tensor = torch.FloatTensor(next_state)
            next_q_values = dqn(next_state_tensor)
            target_q_value = reward + GAMMA * torch.max(next_q_values)

            # Compute current Q value
            state_tensor = torch.FloatTensor(state)
            current_q_value = dqn(state_tensor)[action]

            # Compute loss
            loss = criterion(current_q_value, target_q_value)

            if iteration % 50 == 49:  # log data every 50 iterations
                writer.add_scalar('reward', reward, episode * iteration)
                writer.add_scalar('loss', loss, episode * iteration)
                prop_mass = next_state[12]
                energy_lvl = next_state[13]
                data_left = next_state[14]

                writer.add_scalar('Propellant Mass', prop_mass, episode * iteration)
                writer.add_scalar('Energy Level', energy_lvl, episode * iteration)
                writer.add_scalar('Data Left', data_left, episode * iteration)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            iteration += 1
        current_dir = os.getcwd()
        time_stamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")

    writer.close()



    # env.init_renderer(render_mode="human")
    # env.render()

    results = {"Energy used": env.en_used, "Propellent used": env.prop_used, "Reward": reward,
               "Data Transferred": env.data_sent}
    return results

def save_model(dqn, optimizer=None, epoch=0):
    current_dir = os.getcwd()
    time_stamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")
    if optimizer:
        torch.save(dqn.state_dict(), f"{current_dir}\\all_data_send_model_{time_stamp}.pt")
    else:
        state = {
            'epoch': epoch,
            'state_dict': dqn.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f"{current_dir}\\all_data_send_model_{time_stamp}.pt")

def load_model(path, env):
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    main()
    print("Finished")
