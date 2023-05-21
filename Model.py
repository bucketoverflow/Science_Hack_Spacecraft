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
        x = torch.cat([x, a], dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main(actor=None, critic=None):
    # Initialize environment
    env = Spacecraft()
    env.reset()

    # Hyperparameters
    EPISODES = 800
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 0.01
    GAMMA = 0.999
    LR = 0.001
    TAU = 0.1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    act_space = 4

    if actor and critic:
        actor = actor.to(device)
        critic = critic.to(device)
    else:
        obs_space = env.observation_space.shape[0]
        actor = Actor(obs_space, act_space).to(device)
        critic = Critic(obs_space, act_space).to(device)

    target_actor = Actor(env.observation_space.shape[0], act_space).to(device)
    target_critic = Critic(env.observation_space.shape[0], act_space).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)
    criterion = nn.MSELoss()

    writer = SummaryWriter(
        f'runs/spacecraft_experiment_1_{str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")}')

    iteration = 0

    for episode in range(EPISODES):
        print(f"running episode {episode} from {EPISODES}")
        eps = 0.05 + (EPS_START - 0.05) * np.exp(-1 * episode / EPISODES)
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        done = False
        truncated = False

        while not done and not truncated:
            #actor.eval()
            with torch.no_grad():
                action = actor(state)
            actor.train()

            noise = torch.normal(mean=0., std=0.2, size=action.shape).to(device)
            action = (action + noise).clamp(0, 3)

            act_res = action.cpu().numpy()
            next_state, reward, done, terminated = env.step(np.argmax(act_res))
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
            prop_mass = next_state[15]
            energy_lvl = next_state[16]
            data_left = next_state[17]
            orb_com = next_state[0]
            orb_cbs = next_state[1]
            orb_diff = next_state[2]

            if iteration % 50 == 49:  # log data every 50 iterations
                writer.add_scalar('reward', reward, episode * iteration)
                writer.add_scalar('loss', actor_loss, episode * iteration)
                writer.add_scalar('Propellant Mass', prop_mass, episode * iteration)
                writer.add_scalar('Energy Level', energy_lvl, episode * iteration)
                writer.add_scalar('Data Left', data_left, episode * iteration)
                writer.add_scalar('Tau', TAU, episode * iteration)
                writer.add_scalar('Orbit Communicator', orb_com, episode * iteration)
                writer.add_scalar('Orbit Observer', orb_cbs, episode * iteration)
                writer.add_scalar('Orbit difference', orb_diff, episode * iteration)

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))
            if data_left <= 10:
                track_best_model(actor, env)
                # save_model(dqn)
                writer.add_scalar('reward', reward, episode * iteration)
                writer.add_scalar('actor_loss', actor_loss, episode * iteration)
                prop_mass = next_state[15]
                energy_lvl = next_state[16]
                data_left = next_state[17]

                writer.add_scalar('Propellant Mass', prop_mass, episode * iteration)
                writer.add_scalar('Energy Level', energy_lvl, episode * iteration)
                writer.add_scalar('Data Left', data_left, episode * iteration)
                writer.add_scalar('Tau', TAU, episode * iteration)

            state = next_state
            iteration += 1
        current_dir = os.getcwd()
        time_stamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_").replace(".", "_")
        if env.get_reward() > 4:
            save_model(actor, optimizer_actor, episode)
            print(reward)
    writer.close()

    save_model(actor, optimizer_actor, EPISODES)

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


def load_model(path, env, for_training=True):
    state = torch.load(path)

    if for_training:
        model = Actor(env.observation_space.shape[0], env.action_space.n)
        model.load_state_dict(state['state_dict'])

        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(state['optimizer'])

        return model, optimizer
    else:
        model = Actor(env.observation_space.shape[0], env.action_space.n)
        model.load_state_dict(state['state_dict'])
        model.eval()

        return model, None


def track_best_model(dqn, env):
    pass
    # bestmodeldata.append((
    #     env.data_sent,
    #     dqn.state_dict(),
    #     env.orbit_propagator.positions_obs,
    #     env.orbit_propagator.positions_com,
    #     env.en_used, env.prop_used))


if __name__ == "__main__":
    env = Spacecraft()
    # model = load_model(
    # "/Users/benedikt/Desktop/environment/Klon\model_2023-05-2022_23_33_408626.pt", env)
    Result = main()
    print("Finished")

print("finished")
