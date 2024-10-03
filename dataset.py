import envs
import myeplus
import reward
import policy
import pandas
from collections import namedtuple
import pickle
import lzma
import typing

# Our environment configuration (building, weather, variables)
config = envs.crawlspace()


# how we compute our reward
compute_reward_fn = reward.day_night_reward(
    reward.range_reward(20, 23), reward.range_reward(15, 30)
)
# reward.combine(
#     reward.range_reward(15, 20),
#     reward.energy_reward(),
# )

RunResult = namedtuple("RunResult", ["observations", "actions", "rewards"])


def flatten_observation(obs) -> tuple[typing.Any, typing.Any]:
    keys = []
    values = []
    for zone, v in obs["temperature"].items():
        keys.append("temp_" + zone)
        values.append(v.inner)
    for name, v in obs["time"].items():
        keys.append("time_" + name)
        values.append(v.inner)

    return (keys, values)


def rollout_policy(env, policy):
    observations = []
    actions = []
    rewards = []

    env.start()
    obs, done = env.get_obs()
    while not done:
        keys, flattened = flatten_observation(obs)
        # print(f"keys: {keys}")
        observations.append(flattened)

        action = policy(obs)
        actions.append(action)

        reward = compute_reward_fn(obs)
        rewards.append(reward)

        obs, done = env.step(action)

    return RunResult(
        pandas.DataFrame(observations, columns=keys),
        pandas.DataFrame(actions),
        pandas.DataFrame(rewards, columns=["reward"]),
    )


def day_night(name="day_night"):
    compute_action_fn = policy.day_night(
        policy.const(heating_sch=18, cooling_sch=20),
        policy.const(heating_sch=23, cooling_sch=30),
    )

    env = myeplus.EnergyPlus(*config, max_steps=1_000_000)

    result = rollout_policy(env, compute_action_fn)

    pickle.dump(result, lzma.open(f"pickles/{name}.pkl.xz", "wb"))


def random_walk(name="random_walk"):
    heating_range = (0, 20)
    cooling_range = (20, 40)

    env = myeplus.EnergyPlus(*config, max_steps=1_000_000)

    compute_action_fn = policy.chain(
        policy.combine(
            policy.random_walk("cooling_sch", *cooling_range),
            policy.random_walk("heating_sch", *heating_range),
        ),
        lambda obs: policy.force_actuator_smaller(
            obs,
            "heating_sch",
            "cooling_sch",
        ),
    )
    result = rollout_policy(env, compute_action_fn)
    pickle.dump(result, lzma.open(f"pickles/{name}.pkl.xz", "wb"))


def test():
    env = myeplus.EnergyPlus(*config)
    env.start()
    return env


if __name__ == "__main__":
    # random_walk("random_walk_1")
    # random_walk("random_walk_2")
    day_night("day_night_2")
