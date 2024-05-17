import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.common import get_ids
from sinergym.envs import EplusEnv
import profile
import os

# env = EplusEnv(
#     os.getcwd() +  "/converted.json", "COL_Bogota.802220_IWEC.epw",
#     reward_kwargs={
#         "temperature_variables": [],
#         "energy_variables": [],
#         "range_comfort_winter": [],
#         "range_comfort_summer": [],
#     })
env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1')

obs, info = env.reset()

s = env.unwrapped
