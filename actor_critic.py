import jax
import jax.numpy as np
import jax.random as random
import jax.nn as nn
import gymnasium as gym
from functools import partial
from dataclasses import dataclass
import optax
import trajectory
import pickle
import time
import typing


def once(func):
    ran = False

    def f(*args, **kwargs):
        nonlocal ran
        if not ran:
            ran = True
            func(*args, **kwargs)

    return f


def init_params(key, i, o, layers):
    ks = random.split(key, layers + 2)
    params = []
    for k in ks[1:-1]:
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
def clog(x):
    res = np.log(np.maximum(x, 0.0001))
    return res


@jax.jit
def forward_actor(params, x):
    W, b = params[0]
    x = np.sin(W @ x + b)
    for (W, b) in params[1:-1]:
        x = nn.sigmoid(W @ x + b)
    W, b = params[-1]
    x = nn.softmax(W @ x + b)
    return x


@jax.jit
def forward_critic(params, x):
    W, b = params[0]
    x = np.sin(W @ x + b)
    for (W, b) in params[1:-1]:
        x = nn.sigmoid(W @ x + b)
    W, b = params[-1]
    x = W @ x + b
    return np.reshape(x, ())


def advantage_single(params, s1, a, advantage, weight):
    v = forward_actor(params, s1)[a]
    return -weight * clog(v) * advantage


def advantage_sum(params, s1, a, advantage, weight):
    return jax.vmap(advantage_single, (None, 0, 0, 0, None))(
        params, s1, a, advantage, weight
    ).sum()


def update_weights_actor(
    key,
    actor_params,
    critic_params,
    trajectories: list[trajectory.Trajectory],
    gamma=0.95,
    learning_rate=1e-3,
):
    print("actor weights update step")
    # optimizer = optax.adam(learning_rate)
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(actor_params)

    s1, ap, a, r, s2 = trajectory.merge_trajectories(trajectories, gamma)
    expected_value = jax.vmap(forward_critic, (None, 0))(critic_params, s1)
    advantage = r - expected_value
    past_prob = jax.vmap(clog)(ap).sum()
    new_prob = jax.vmap((lambda s, a: clog(forward_actor(actor_params, s)[a])), (0, 0))(
        s1, a
    ).sum()
    logweight = 0.0  # min(new_prob - past_prob, 2)
    grad = jax.grad(advantage_sum)(actor_params, s1, a, advantage, np.exp(logweight))
    updates, opt_state = optimizer.update(grad, opt_state)
    actor_params = optax.apply_updates(actor_params, updates)
    return actor_params


def compute_value_loss_single(critic_params, state, discounted_reward):
    return (forward_critic(critic_params, state) - discounted_reward) ** 2


def compute_value_loss(critic_params, state, discounted_reward):
    return jax.vmap(compute_value_loss_single, (None, 0, 0))(
        critic_params, state, discounted_reward
    ).sum()


def update_weights_critic(
    key, critic_params, trajectories, gamma=0.95, learning_rate=1e-3
):

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(critic_params)
    perms = random.permutation(key, len(trajectories))
    sum_of_losses = 0.0
    for i, traj in enumerate(trajectories[i] for i in perms):
        print(f"\rupdating critic ({(i+1)/len(trajectories):0.2f})", end="")
        s1, _, a, _, s2 = traj.get_arrays()
        r = traj.rewards_to_go(gamma)
        val, grad = jax.value_and_grad(compute_value_loss)(critic_params, s1, r)
        updates, opt_state = optimizer.update(grad, opt_state)
        critic_params = optax.apply_updates(critic_params, updates)
        sum_of_losses += val / traj.len()
    sum_of_losses /= len(trajectories)
    print(f" mean critic loss per step: {sum_of_losses}")
    return critic_params, sum_of_losses


def naturals():
    i = 0
    while True:
        yield i
        i += 1


def until_convergence(f, epsilon=0.1):
    x = f()
    x_last = x
    while True:
        x_new = f()
        if x_new < x_last and (x_last - x_new) / x < epsilon:
            break
        x_last = x_new


def dothing():
    key = random.key(123)

    (
        actor_params_key,
        critic_params_key,
        action_key,
        env_seed,
        perm_key,
    ) = random.split(key, 5)

    env = gym.make("LunarLander-v2")  # , render_mode="human")

    actor_params = init_params(actor_params_key, 8, 4, 5)
    critic_params = init_params(critic_params_key, 8, 1, 4)

    def run_trajectories(n):
        nonlocal env_seed, action_key
        trajectories = []

        for i in range(n):
            print(f"\rgenerating trajectories ({(i+1)/n:0.2f})", end="")
            env_seed, k = random.split(env_seed)
            observation, info = env.reset(seed=int(random.randint(k, (), 0, 1000)))
            traj = trajectory.Trajectory()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action_probs = forward_actor(actor_params, observation)
                k, action_key = random.split(action_key)
                action = int(random.categorical(k, action_probs))
                new_obs, reward, terminated, truncated, info = env.step(action)
                traj.add_transition(
                    observation, action_probs[action], action, reward, new_obs
                )
                observation = new_obs
            trajectories.append(traj)
        print()
        return trajectories

    batch_size = 200
    trajectories = []
    gamma = 0.99
    for epoch in naturals():
        print(f"epoch {epoch}")

        if epoch % 10 == 0:
            with open(f"pickles/model-{int(time.time())}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "actor_params": actor_params,
                        "critic_params": critic_params,
                    },
                    f,
                )
        trajectories = run_trajectories(batch_size)

        rewards = np.array([traj.sum_of_rewards() for traj in trajectories])

        print(f"reward of last batch: {rewards.mean():.1f}±{rewards.std():.2f}")
        k1, k2, perm_key = random.split(perm_key, 3)

        def update():
            nonlocal critic_params
            nonlocal k2

            k, k2 = random.split(k2)
            critic_params, loss = update_weights_critic(
                k,
                critic_params,
                trajectories,
                gamma,
                learning_rate=1e-3,
            )
            return loss

        until_convergence(update, epsilon=5e-3)

        actor_params = update_weights_actor(
            k1,
            actor_params,
            critic_params,
            trajectories,
            gamma,
            learning_rate=1e-4,
        )


def test():
    filename = "pickles/model-1717470497.pkl"
    with open(filename, "rb") as f:
        params = pickle.load(f)["actor_params"]
    env = gym.make("LunarLander-v2", render_mode="human")
    key = random.key(123)
    action_key, seed_key = random.split(key)

    while True:
        terminated = False
        truncated = False
        k, seed_key = random.split(seed_key)
        obs, info = env.reset(seed=int(random.randint(k, (), 0, 1000)))
        while not (terminated or truncated):
            action_probs = forward_actor(params, obs)
            k, action_key = random.split(action_key)
            action = int(random.categorical(k, action_probs))
            new_obs, reward, terminated, truncated, info = env.step(action)
            obs = new_obs


if __name__ == "__main__":
    dothing()
