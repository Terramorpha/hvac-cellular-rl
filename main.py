from dataclasses import dataclass
import flax
import flax.linen as linen
import functools
import gymnasium as gym
import jax
import jax.numpy as np
import jax.random as random
import numpy
import optax
import os
import pathlib
import pickle
import sinergym
import split
import time
import trajectory
import typing
import utils


class Q(linen.Module):
    @linen.compact
    def __call__(self, observation, action):

        x = np.concatenate((observation, action), 0)
        for n in [20, 20, 20]:
            x = linen.Dense(features=n)(x)
            x = linen.leaky_relu(x)
        x = linen.Dense(features=1)(x)
        return np.reshape(x, ())


class MultiQ(linen.Module):
    n_qs: int

    @linen.compact
    def __call__(self, observation, action):
        return min([Q()(observation, action) for _ in range(self.n_qs)])


class Policy(linen.Module):
    out_dims: int

    @linen.compact
    def __call__(self, x):
        for n in [20, 20, 20]:
            x = linen.Dense(features=n)(x)
            x = linen.leaky_relu(x)

        x = linen.Dense(features=self.out_dims)(x)
        return x


def generate_trajectories(env, key: jax.dtypes.prng_key, policy, n=10):
    trajectories = []
    for i in range(n):
        k = random.fold_in(key, i)
        trajectories.append(run_trajectory(env, k, policy))
    return trajectories


@dataclass
class Model:
    q: Q
    q_params: typing.Any
    pi: Policy
    pi_params: typing.Any

    def get_pi_param_func(self, env):
        scale = space_scale(env.action_space)

        def policy(params, state):
            x = self.pi.apply(params, state)
            return scale(x)

        return policy

    def get_pi_func(self, env):
        return functools.partial(self.get_pi_param_func(env), self.pi_params)

    def get_q_func(self):
        return functools.partial(self.q.apply, self.q_params)

    def many_trajectories(self, env, key: jax.dtypes.prng_key, n=10):
        policy = jax.jit(self.get_pi_func(env))
        return generate_trajectories(env, key, policy, n=n)

    def train_pi_trajectories(
        self,
        env: gym.Env,
        key: jax.dtypes.prng_key,
        trajectories: list[trajectory.Trajectory],
        learning_rate=1e-3,
        epochs=100,
        until_convergence: bool = False,
        convergence_epsilon=100,
        batch_size=128,
    ):

        print("updating the pi function")
        q_function = self.get_q_func()
        pi_func = jax.jit(self.get_pi_param_func(env))
        print(f"learning rate: {learning_rate}")
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.pi_params)

        def value_func(pi_params, observation):
            action = pi_func(pi_params, observation)
            return q_function(observation, action)

        vectorized_value = jax.vmap(value_func, (None, 0))

        def loss(params, obs):
            # Negatif parce que c'est une neg loss
            return -vectorized_value(params, obs).sum()

        gradientized_value = jax.jit(jax.value_and_grad(loss))

        s1, p, a, r, s2 = trajectory.merge_trajectories(
            trajectories, 0.0
        )  # Ici, le gamma fait rien

        def trainstep():
            nonlocal opt_state
            # perms_key = random.fold_in(key, epoch)
            val, grad = gradientized_value(self.pi_params, s1)
            updates, opt_state = optimizer.update(grad, opt_state)
            self.pi_params = optax.apply_updates(self.pi_params, updates)
            print(f"val (smaller is better): {val}")
            return val

        if until_convergence:
            print(f"training until convergence (ε = {convergence_epsilon})")
            utils.until_convergence(trainstep, convergence_epsilon)
        else:
            with utils.Progress("epoch", range(epochs)) as p:
                for epoch in p:
                    trainstep()

    def train_q_trajectories(
        self,
        env: gym.Env,
        key: jax.dtypes.prng_key,
        trajectories: list[trajectory.Trajectory],
        learning_rate=1e-4,
        gamma=0.95,
        epochs=100,
        policy=None,
    ):
        if policy is None:
            policy = self.get_pi_func(env)
        print("updating the q function")
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.q_params)

        q_apply = self.q.apply

        vectorized = jax.vmap(
            lambda q_params, s1, a1, p, r, s2: bellman_error(
                q_apply(q_params, s1, a1),
                s1,
                a1,
                p,
                r,
                s2,
                q_apply(q_params, s1, policy(s2)),
            ),
            (None, 0, 0, 0, 0, 0),
        )
        gradientized = jax.jit(
            jax.value_and_grad(
                lambda params, s1, a, p, r, s2: vectorized(
                    params, s1, a, p, r, s2
                ).sum()
                / len(trajectories)
            )
        )

        for epoch in range(epochs):
            print(f"epoch {epoch}/{epochs}")
            perms_key = random.fold_in(key, epoch)
            # on permute les trajectoires à chaque fois
            perms = random.permutation(perms_key, len(trajectories))

            losses = 0.0
            with utils.Progress("q update", (trajectories[i] for i in perms)) as t:
                for traj in t:
                    s1, p, a, r, s2 = traj.get_arrays()

                    val, grad = gradientized(self.q_params, s1, a, p, r, s2)
                    losses += val
                    updates, opt_state = optimizer.update(grad, opt_state)
                    self.q_params = optax.apply_updates(self.q_params, updates)
            print(f" loss: {losses/len(trajectories)}")
        print()


def init_model(key: jax.dtypes.prng_key, obs_sample, action_sample):
    k1, k2 = random.split(key)

    policy_model = Policy(out_dims=2)
    policy_params = policy_model.init(k1, obs_sample)

    q_model = Q()
    q_params = q_model.init(k2, obs_sample, action_sample)

    return Model(
        q=q_model,
        q_params=q_params,
        pi=policy_model,
        pi_params=policy_params,
    )


def run_trajectory(env, key: jax.dtypes.prng_key, policy: typing.Callable):
    env_key, action_key = random.split(key)
    s = int(random.randint(env_key, (), 0, 255))
    obs, info = env.reset(seed=s)
    truncated = False
    terminated = False

    traj = trajectory.Trajectory()
    i = 0
    while not (truncated or terminated):
        # on ignore l'observation
        action = policy(obs)
        space = env.action_space
        noise = random.normal(random.fold_in(action_key, i), env.action_space.shape)
        action += noise
        clipped_action = np.minimum(np.maximum(space.low, action), space.high)
        # parce que le pyenergyplus.common.is_number accepte pas les Array 0d.
        clipped_action = numpy.array(clipped_action)
        new_obs, r, truncated, terminated, info = env.step(clipped_action)
        traj.add_transition(obs, 1.0, action, r, new_obs)
        obs = new_obs
        i += 1
    return traj


def space_loss(point, space: gym.spaces.Box):
    """Compute a loss corresponding to ℓ² distance to the nearest boundary point of a box."""
    v = (point < space.low) * (point - space.low) + (point > space.high) * (
        point - space.high
    )
    v = v**2
    return 100000 * v.sum()


def space_scale(space: gym.spaces.Box):
    """Using a sigmoid activation function, map ℝⁿ into the given box."""
    lo = space.low
    hi = space.high

    def scale(x):
        x = linen.sigmoid(1e-3 * x / (hi - lo))
        return (hi - lo) * x + lo

    return scale


def bellman_error(qs1a1, s1, a1, p, r, s2, qs2a2, gamma=0.95):

    # La bellman equation pour Q:

    # Q(s, a) = r + γ * argmax_a' Q(s', a')

    # Ici, a est l'action qu'on sait qu'on a pris et le argmax_a' Q(s', a') va
    # être calculé par `policy`.

    error = qs1a1 - (r + gamma * qs2a2)
    return error**2


def cache(name: str | None = None):
    def func(thunk):
        nonlocal name
        if name is None:
            name = thunk.__name__
        p = pathlib.Path("./pickles") / pathlib.Path(name + ".pkl")
        if not p.is_file():
            val = thunk()
            with open(p, "wb") as f:
                pickle.dump(val, f)
                return val
        else:
            with open(p, "rb") as f:
                val = pickle.load(f)
                return val

    return func


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


def main():
    N_things = 10

    env = gym.make("Eplus-demo-v1")
    key = random.key(12345)
    model_key, traj_key, pi_key, q_key, key = random.split(key, 5)
    # Pour obtenir la shape
    obs_sample = env.observation_space.sample()
    action_sample = env.action_space.sample()

    def random_policy(state):
        return env.action_space.sample()

    @cache()
    def random_trajectories():
        return generate_trajectories(
            env, random.fold_in(traj_key, -1), random_policy, 100
        )

    mr = trajectory.mean_reward(random_trajectories)
    print(f"mean reward of random policy: {mr}")

    @cache()
    def initial_models_with_q():
        mods = []

        for i in range(N_things):

            mod = init_model(random.fold_in(model_key, i), obs_sample, action_sample)

            mod.train_q_trajectories(
                env,
                random.fold_in(random.fold_in(q_key, -1), i),
                random_trajectories,
                epochs=100,
                learning_rate=1e-3,
                policy=random_policy,
                gamma=0.95,
            )
            mods.append(mod)
        return mods

    @cache()
    def initial_models_with_pi():
        mods = []
        for i in range(N_things):
            initial_models_with_q[i].train_pi_trajectories(
                env,
                random.fold_in(random.fold_in(pi_key, -1), i),
                random_trajectories,
                learning_rate=2e-2,
                until_convergence=True,
            )
            mods.append(initial_models_with_q[i])

        return mods

    all_trajectories = random_trajectories
    trajectories = random_trajectories

    models = initial_models_with_pi

    for epoch in range(100):

        @cache(name=f"trajectories_{epoch}")
        def new_trajectories():
            trajs = []
            for i in range(N_things):
                trajs += models[i].many_trajectories(
                    env, random.fold_in(traj_key, i), n=50
                )
            return trajs

        mr = trajectory.mean_reward(new_trajectories)
        print(f"total mean reward: {mr}")
        # trajectories += new_trajectories
        trajectories = new_trajectories
        all_trajectories += new_trajectories

        @cache(name=f"q_trained_{epoch}")
        def q_trained():
            mods = []
            for i in range(N_things):
                models[i].train_q_trajectories(
                    env,
                    random.fold_in(q_key, i),
                    all_trajectories,
                    epochs=20,
                    learning_rate=1e-4,
                )
                mods.append(models[i])
            return mods

        @cache(name=f"pi_trained_{epoch}")
        def pi_trained():
            mods = []
            for i in range(N_things):
                q_trained[i].train_pi_trajectories(
                    env,
                    random.fold_in(pi_key, i),
                    all_trajectories,
                    learning_rate=1e-3,
                    epochs=100,
                )
                mods.append(q_trained[i])
            return mods

        model = pi_trained


if __name__ == "__main__":
    main()
