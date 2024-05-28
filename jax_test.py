import jax
import jax.numpy as np
import jax.random as rand
import jax.tree_util
import jax.nn as nn
import gymnasium as gym
from typing import Callable
from functools import partial
from dataclasses import dataclass

# env = gym.make("LunarLander-v2", render_mode="human")

# print(env.action_space)
# print(env.observation_space)
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()


def identity(x):
    return x


def init_params(key, i, o):
    k1, k2, k3, k4 = rand.split(key, 4)
    params = [
        (rand.normal(k1, (i, i)), rand.normal(k1, (i,))),
        (rand.normal(k2, (i, i)), rand.normal(k2, (i,))),
        (rand.normal(k3, (o, i)), rand.normal(k3, (o,))),
    ]
    return params, k4


def predict(params, x):
    for (W, b) in params[:-1]:
        x = nn.relu(W @ x + b)
    W, b = params[-1]
    x = nn.softmax(W @ x + b)
    return x


@jax.jit
def clog(x):
    res = np.log(np.maximum(x, 0.0001))
    return res


@jax.jit
def J(params, trajectory: list, gamma=0.9):
    s = 0.0
    discounted_rewards = [trajectory[-1][2]]
    for (_, _, r, _) in trajectory[:-1][::-1]:
        discounted_rewards.append(discounted_rewards[-1] * gamma + r)
    discounted_rewards.reverse()

    def calc(x, r):
        (s1, a, _, s2) = x
        return clog(predict(params, s1)[a]) * r

    s = 0.0
    for (s1, a, _, s2), r in zip(trajectory, discounted_rewards):
        val = predict(params, s1)[a]
        s += clog(val) * r

    return s


@jax.jit
def compute_grad(params, trajectory):
    return jax.grad(J)(params, trajectory)


key = rand.key(0)
params, key = init_params(key, 8, 4)


epochs = 5

for epoch in range(epochs):
    print("epoch", epoch)
    trajectories = []

    env = gym.make("LunarLander-v2")

    N = 20
    for i in range(N):
        terminated = False
        truncated = False
        current_observation, info = env.reset(
            seed=int(rand.choice(key, np.array(range(100))))
        )
        trajectory = []

        while not (terminated or truncated):
            action_prob = predict(params, current_observation)
            used_key, key = rand.split(key)
            choice = int(rand.categorical(used_key, action_prob))
            new_observation, reward, terminated, truncated, info = env.step(choice)
            trajectory.append((current_observation, choice, reward, new_observation))
            current_observation = new_observation
        trajectories.append(trajectory)

    # On veut backpropagate sur une trajectoire

    eta = 1e-5
    before = J(params, trajectories[0])
    newparams = params
    for traj in trajectories:
        grad = compute_grad(params, traj)
        newparams = jax.tree_util.tree_map(lambda a, b: a + eta * b, params, grad)
    after = J(params, trajectories[0])
    params = newparams
    print(f"after: {after}")
    print(f"after - before: {after - before}")


def test_policy():
    env = gym.make("LunarLander-v2", render_mode="human")
    current_observation, info = env.reset(seed=42)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action_prob = predict(params, current_observation)
        used_key, key = rand.split(key)
        choice = int(rand.categorical(used_key, action_prob))
        new_observation, reward, terminated, truncated, info = env.step(choice)
