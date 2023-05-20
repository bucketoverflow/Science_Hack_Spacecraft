import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    from itertools import count

    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)

    env = Spacecraft(render_mode="human")
    env = Spacecraft()
    env.reset()
    for t in count():
        observation, reward, terminated, truncated, _ = env.step(0)
        print(env.get_state())
        done = terminated or truncated
        if done:
            break

