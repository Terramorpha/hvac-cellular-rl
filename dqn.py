import jax
import jax.numpy as np
import jax.random as random
import jax.nn as nn
import gymnasium as gym
from functools import partial
from dataclasses import dataclass
import optax
from trajectory import Trajectory
import pickle
import time


def once(func):
    ran = False

    def f(*args, **kwargs):
        nonlocal ran
        if not ran:
            ran = True
            func(*args, **kwargs)

    return f


def init_params(key, i, o, layers):
    ks = random.split(key, layers + 1)
    params = []
    for k in ks[:-1]:
        k1, k2 = random.split(key)
        params.append(
            (
                random.normal(k1, (i, i)),
                random.normal(k2, (i,)),
            )
        )
    k1, k2 = random.split(key)
    params.append(
        (
            random.normal(k1, (o, i)),
            random.normal(k2, (o,)),
        )
    )
    return params


@jax.jit
def forward(params, x):
    for (W, b) in params[:-1]:
        x = nn.sigmoid(W @ x + b)
    W, b = params[-1]
    x = W @ x + b
    return x


@jax.jit
def compute_bellman_loss_single(params, s1, a, r1, discounted_r2):
    q_values = forward(params, s1)
    q = q_values[a]
    return (q - (r1 + discounted_r2)) ** 2


def compute_bellman_loss(key, target_params, params, traj, gamma):
    s1, a, r, s2 = traj.get_arrays()
    # perms = random.permutation(key, len(a))
    # s1 = s1[perms, :]
    # a = a[perms]
    # r = r[perms]
    # s2 = s2[perms]
    discounted_r2s = jax.vmap(lambda s2: gamma * forward(target_params, s2).max())(s2)
    func = jax.vmap(compute_bellman_loss_single, (None, 0, 0, 0, 0))
    return func(params, s1, a, r, discounted_r2s).sum()


def update_weights(
    key, target_params, params, trajectories, gamma=0.95, learning_rate=1e-3
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    perms = random.permutation(key, len(trajectories))
    for i, traj in enumerate([trajectories[i] for i in perms]):
        print(f"\rupdating ({(i+1)/len(trajectories):0.2f})", end="")

        grad = jax.grad(compute_bellman_loss, argnums=2)(
            key, target_params, params, traj, gamma
        )
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    print()
    return params


@once
def save_1000(params):
    t = int(time.time())
    with open(f"pickles/{t}-1000.pkl", "wb") as f:
        pickle.dump(params, f)


@once
def save_400(params):
    t = int(time.time())
    with open(f"pickles/{t}-400.pkl", "wb") as f:
        pickle.dump(params, f)


@once
def save_200(params):
    t = int(time.time())
    with open(f"pickles/{t}-200.pkl", "wb") as f:
        pickle.dump(params, f)


@once
def save_150(params):
    t = int(time.time())
    with open(f"pickles/{t}-150.pkl", "wb") as f:
        pickle.dump(params, f)


def do_thing():
    key = random.key(1)
    (params_key, action_key, env_seed, perm_key) = random.split(key, 4)

    env = gym.make("LunarLander-v2")  # , render_mode="human")

    epsilon = 0.2

    params = init_params(params_key, 8, 4, 5)
    target_params = params

    trajectories = []

    batch_size = 10

    memory_length = 1000

    for epoch in range(200):
        print(f"epoch {epoch}")
        if epoch % 10 == 0:
            target_params = params

        if len(trajectories) >= memory_length:
            trajectories = trajectories[batch_size:]
        for _ in range(batch_size):
            env_seed, k = random.split(env_seed)
            observation, info = env.reset(
                seed=int(random.randint(env_seed, (), 0, 1000))
            )
            trajectory = Trajectory()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                q_values = forward(params, observation)
                k, action_key = random.split(action_key)
                r = random.uniform(k)
                if r < epsilon:
                    k, action_key = random.split(action_key)
                    action = int(random.randint(k, (), 0, 4))
                else:
                    action = int(q_values.argmax())
                new_obs, reward, terminated, truncated, info = env.step(action)
                trajectory.add_transition(observation, action, reward, new_obs)
                observation = new_obs
            trajectories.append(trajectory)

        m = sum(traj.reward() for traj in trajectories) / len(trajectories)
        if m > -1000:
            save_1000(params)
        if m > -400:
            save_400(params)
        if m > -200:
            save_200(params)
        if m > -150:
            save_150(params)
        print(f"mean reward: {m}")
        k, perm_key = random.split(perm_key)
        params = update_weights(
            perm_key,
            target_params,
            params,
            trajectories,
            gamma=0.85,
            learning_rate=1e-3,
        )


def test_params(params):
    env = gym.make("LunarLander-v2", render_mode="human")

    action_key = random.key(0)

    for i in range(100):
        terminated = False
        truncated = False
        observation, info = env.reset(seed=i)
        while not (terminated or truncated):
            q_values = forward(params, observation)
            k, action_key = random.split(action_key)
            action = int(q_values.argmax())
            new_obs, reward, terminated, truncated, info = env.step(action)
            observation = new_obs


def test():
    with open("pickles/1717110772-150.pkl", "rb") as f:
        params = pickle.load(f)
    test_params(params)


test()
