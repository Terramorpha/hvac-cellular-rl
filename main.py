import gymnasium as gym
import sinergym
import split
import jax
import jax.numpy as np
import jax.random as random
import trajectory
import flax
import flax.linen as linen
import typing
import functools
import numpy
import optax
import time
import pickle
import utils


class Q(linen.Module):
    @linen.compact
    def __call__(self, observation, action):
        x = np.concatenate((observation, action), 0)
        for n in [20, 30, 40]:
            x = linen.Dense(features=n)(x)
            x = linen.relu(x)
        x = linen.Dense(features=1)(x)
        return np.reshape(x, ())


class Policy(linen.Module):
    out_dims: int

    @linen.compact
    def __call__(self, x):
        x = linen.Dense(features=10)(x)
        x = np.sin(x)

        for n in [10, 15, 20]:
            x = linen.Dense(features=n)(x)
            x = linen.relu(x)

        x = linen.Dense(features=self.out_dims)(x)
        return x


def run_trajectory(env, key: jax.dtypes.prng_key, policy: typing.Callable):
    env_key, action_key = random.split(key)
    s = int(random.randint(env_key, (), 0, 255))
    obs, info = env.reset(seed=s)
    truncated = False
    terminated = False

    traj = trajectory.Trajectory()

    while not (truncated or terminated):
        # on ignore l'observation
        action = policy(obs)
        space = env.action_space

        clipped_action = np.minimum(np.maximum(space.low, action), space.high)
        # parce que le pyenergyplus.common.is_number accepte pas les Array 0d.
        clipped_action = numpy.array(clipped_action)
        new_obs, r, truncated, terminated, info = env.step(clipped_action)
        traj.add_transition(obs, 1.0, action, r, new_obs)
        obs = new_obs
    return traj


def many_trajectories(env, key: jax.dtypes.prng_key, policy: typing.Callable, n=10):
    trajectories = []
    for i in range(n):
        trajectories.append(run_trajectory(env, key, policy))
    return trajectories


def space_loss(point, space: gym.spaces.Box):
    """Compute a loss corresponding to ℓ² distance to the nearest boundary point of a box."""
    v = (point < space.low) * (point - space.low) + (point > space.high) * (
        point - space.high
    )
    v = v**2
    return v.sum()


def bellman_error(qs1a1, s1, a1, p, r, s2, qs2a2, gamma=0.95):

    # La bellman equation pour Q:

    # Q(s, a) = r + γ * argmax_a' Q(s', a')

    # Ici, a est l'action qu'on sait qu'on a pris et le argmax_a' Q(s', a') va
    # être calculé par `policy`.

    error = qs1a1 - (r + gamma * qs2a2)
    return error**2


def train_q_trajectories(
    key: jax.dtypes.prng_key,
    env: gym.Env,
    q_model,
    q_params,
    policy,
    trajectories: list[trajectory.Trajectory],
    learning_rate=1e-2,
    gamma=0.95,
):
    print("updating the q function")
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(q_params)

    N_epoch = 100

    q_apply = jax.jit(q_model.apply)

    vectorized = jax.vmap(
        lambda q_params, s1, a1, p, r, s2: bellman_error(
            q_apply(q_params, s1, a1),
            s1,
            a1,
            p,
            r,
            s2,
            q_apply(q_params, s1, policy(s2)),
        )
        + space_loss(a1, env.action_space),
        (None, 0, 0, 0, 0, 0),
    )
    gradientized = jax.jit(
        jax.value_and_grad(
            lambda params, s1, a, p, r, s2: vectorized(params, s1, a, p, r, s2).sum()
            / len(trajectories)
        )
    )

    for epoch in range(N_epoch):
        print(f"epoch {epoch}/{N_epoch}")
        perms_key = random.fold_in(key, epoch)
        # on permute les trajectoires à chaque fois
        perms = random.permutation(perms_key, len(trajectories))

        losses = 0.0
        with utils.Progress("trajectories", (trajectories[i] for i in perms)) as t:
            for traj in t:
                s1, p, a, r, s2 = traj.get_arrays()

                val, grad = gradientized(q_params, s1, a, p, r, s2)
                losses += val
                updates, opt_state = optimizer.update(grad, opt_state)
                q_params = optax.apply_updates(q_params, updates)
        print(f" loss: {losses/len(trajectories)}")
    print()
    return q_params


def train_pi_trajectories(
    key: jax.dtypes.prng_key,
    env: gym.Env,
    pi_model,
    pi_params,
    q_function,
    trajectories: list[trajectory.Trajectory],
    learning_rate,
):
    print("updating the pi function")
    apply = jax.jit(pi_model.apply)

    def value(pi_params, observation):
        return q_function(observation, apply(pi_params, observation))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(pi_params)

    vectorized_value = jax.vmap(value, (None, 0))
    # Negatif parce que c'est une neg loss
    def summed_value(params, obs):
        return -vectorized_value(params, obs).sum()

    gradientized_value = jax.value_and_grad(summed_value)

    N_epoch = 100
    for epoch in range(N_epoch):
        print(f"epoch {epoch}/{N_epoch}")
        perms_key = random.fold_in(key, epoch)
        perms = random.permutation(perms_key, len(trajectories))
        losses = 0.0
        with utils.Progress("trajectories", (trajectories[i] for i in perms)) as t:
            for traj in t:
                s1, _, _, _, _ = traj.get_arrays()
                val, grad = gradientized_value(pi_params, s1)
                updates, opt_state = optimizer.update(grad, opt_state)
                pi_params = optax.apply_updates(pi_params, updates)
                losses += val
        print(f" loss: {losses/len(trajectories)}")
    return pi_params


env = gym.make("Eplus-demo-v1")

key = random.key(1234)
policy_key, q_key, key = random.split(key, 3)

# Pour obtenir la shape
obs_sample = env.observation_space.sample()
action_sample = env.action_space.sample()

policy_model = Policy(out_dims=2)
policy_params = policy_model.init(policy_key, obs_sample)

q_model = Q()
q_params = q_model.init(q_key, obs_sample, action_sample)


def save(name, o):
    t = int(time.time())
    with open(f"pickles/{name}-{t}.pkl", "wb") as f:
        pickle.dump(o, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_trajectories_stats(traj: list[trajectory.Trajectory]):
    mean_reward = sum([traj.sum_of_rewards() for traj in traj]) / len(traj)
    print(f"mean reward of current trajectories: {mean_reward}")


def dothing():
    global trajectories, q_params, policy_params
    policy = jax.jit(functools.partial(policy_model.apply, policy_params))
    # trajectories = many_trajectories(env, key, policy, n=50)
    # save("trajectories", trajectories)
    trajectories = load("pickles/trajectories-1717693351.pkl")
    print_trajectories_stats(trajectories)
    # q_params = train_q_trajectories(key, env, q_model, q_params, policy, out, 1e-4)
    q_params = load("pickles/params-1717697845.pkl")
    q_function = jax.jit(functools.partial(q_model.apply, q_params))

    policy_params = train_pi_trajectories(
        key,
        env,
        policy_model,
        policy_params,
        q_function,
        trajectories,
        1e-4,
    )
    save("pi_params", policy_params)
