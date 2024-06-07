import jax
import jax.numpy as np
import jax.random as random
import jax.tree_util
import jax.nn as nn
import gymnasium as gym
from typing import Callable
from functools import partial
from dataclasses import dataclass
from trajectory import Trajectory
import optax
import time
import pickle

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


@jax.jit
def identity(x):
    return x


def init_params(key, i, o, layers):
    k1, k2, k3 = random.split(key, 3)
    params = [
        (random.normal(k1, (i, i)), random.normal(k1, (i,))) for _ in range(layers)
    ] + [
        (random.normal(k3, (o, i)), random.normal(k3, (o,))),
    ]
    return params


@jax.jit
def predict(params, x):
    for (W, b) in params[:-1]:
        x = nn.sigmoid(W @ x + b)
    W, b = params[-1]
    x = nn.softmax(W @ x + b)
    return x


@jax.jit
def clog(x):
    res = np.log(np.maximum(x, 0.0001))
    return res


@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
def J(params, start_state, action, discounted_reward, end_state):
    return clog(predict(params, start_state)[action]) * discounted_reward


def discount(r, gamma):
    discounted_rewards = [r[-1]]
    for r in r[:-1][::-1]:
        discounted_rewards.append(discounted_rewards[-1] * gamma + r)
    discounted_rewards.reverse()
    return np.array(discounted_rewards)


def compute_reward(params, traj: Trajectory, gamma):
    s1, a, r, s2 = traj.get_arrays()
    js = J(params, s1, a, r, s2)

    return js.sum()


def cat_traj(trajectories):
    s1, a, r, s2 = trajectories[0].get_arrays()
    s1s = s1
    actions = a
    rs = discount(r, gamma)
    s2s = s2
    for traj in trajectories[1:]:
        discounted = discount(r, gamma)
        s1s = np.concatenate((s1s, s1), 0)
        actions = np.concatenate((actions, a), 0)
        rs = np.concatenate((rs, discounted), 0)
        s2s = np.concatenate((s2s, s2), 0)
    return s1s, actions, rs, s2s


def reinforce(
    params,
    forward,
    gamma=0.95,
    key=random.key(0),
    learning_rate=1e-3,
    batches=20,
    batch_size=50,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    env = gym.make("LunarLander-v2")
    for batch in range(batches):
        trajectories = []
        for i in range(batch_size):
            terminated = False
            truncated = False
            current_observation, info = env.reset(
                seed=int(random.choice(key, np.array(range(100))))
            )
            trajectory = Trajectory()
            while not (terminated or truncated):
                # print(current_observation)
                action_prob = forward(params, current_observation)
                used_key, key = random.split(key)
                choice = int(random.categorical(used_key, action_prob))
                new_observation, reward, terminated, truncated, info = env.step(choice)
                trajectory.add_transition(
                    current_observation,
                    choice,
                    reward,
                    new_observation,
                )
                current_observation = new_observation
            trajectories.append(trajectory)
        s = sum(sum(traj.reward) for traj in trajectories) / len(trajectories)
        jax.debug.print(
            f"mean reward per trajectory: {s}",
        )
        for i, traj in enumerate(trajectories):
            print(f"\rupdating for trajectory {i/len(trajectories):1.2f}...", end="")
            grad = jax.grad(compute_reward)(params, traj, gamma)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
        print()
