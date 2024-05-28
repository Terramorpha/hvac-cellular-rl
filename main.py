import gymnasium as gym
import sinergym
import split
import jax
import jax.numpy as np

# part = split.FeaturePartition(env)
# separated = part.split_observation(observation)
# gr = part.zones_graph()

# env = gym.make("Eplus-datacenter-mixed-continuous-stochastic-v1")
env = gym.make("Eplus-demo-v1")
obs, info = env.reset()
