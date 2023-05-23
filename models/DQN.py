import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    # definition of the neural network , 1 hidden layer for the moment
    def __init__(self, obs_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, 128)
        # self.fc7 = nn.Linear(128, action_space)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    # using relu all over it
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        return self.fc3(x)

    def save(self, optimizer=None, epoch=0):
        current_dir = os.getcwd()
        time_stamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")
        if optimizer:
            torch.save(self.state_dict(), f"{current_dir}\\all_data_send_model_{time_stamp}.pt")
        else:
            state = {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, f"{current_dir}\\all_data_send_model_{time_stamp}.pt")


def load_model(path, env, for_training=True):
    state = torch.load(path)
    if for_training:
        model = DQN(env.observation_space.shape[0], env.action_space.n)
        model.load_state_dict(state['state_dict'])
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer
    else:
        model = DQN(env.observation_space.shape[0], env.action_space.n)
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model, None
