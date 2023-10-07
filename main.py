import os
import time
import numpy as np
import gym
import torch
from gym import wrappers
from replaybuffer import ReplayBuffer
from agent import TD3
from plot import LivePlot
import pybullet_envs


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




# env_name = "AntBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
env_name = "HumanoidBulletEnv-v0"

if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./plots"):
    os.makedirs("./plots")

# env = gym.make(env_name, render=True) # Good for testing
env = gym.make(env_name)
env.seed(0) # Pick a consistent starting point.

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

stats = {'Returns': [], 'AvgReturns': []}

learning_rate = 0.002

agent = TD3(state_dim, action_dim, max_action, batch_size=100, policy_freq=2,
            discount=0.995, device=device, tau=0.005, policy_noise=0.2, expl_noise=0.2,
            noise_clip=0.5, start_timesteps=1e4, learning_rate=learning_rate, env_name=env_name, lr_decay_factor=0.9999)


stats = agent.train(env, max_timesteps=10e6, stats=stats, batch_identifier=0)


# for i in range(100):
#
#     stats = agent.train(env, max_timesteps=1e5, stats=stats, batch_identifier=i)
#
#     print(f"Completed training iteration {i} with a learning rate of {learning_rate}")
#
#     learning_rate = learning_rate * 0.9
#
#     agent.update_learning_rate(learning_rate=learning_rate)

