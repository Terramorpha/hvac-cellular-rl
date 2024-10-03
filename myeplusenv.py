import typing
import myeplus
import gymnasium
import policy
import numpy as np
import pandas

from gymnasium import spaces


class EnergyPlusEnv(gymnasium.Env):
    metadata = {"render_modes": []}
    make_energyplus: typing.Callable[[], myeplus.EnergyPlus]

    def action_transform(self, v):
        # print(f"got {v}")
        return policy.force_actuator_smaller(
            {
                "cooling_sch": v[0],
                "heating_sch": v[1],
            },
            "heating_sch",
            "cooling_sch",
        )

    def __init__(
        self,
        make_energyplus: typing.Callable[[], myeplus.EnergyPlus],
        reward_fn,
    ):
        super(EnergyPlusEnv, self).__init__()
        self.make_energyplus = make_energyplus
        self.reward_fn = reward_fn

        self.observation_space = spaces.Box(
            low=np.concatenate((np.zeros(24), -np.ones(6))),
            high=np.concatenate((40 * np.ones(24), np.ones(6))),
            shape=(30,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=40.0, shape=(2,), dtype=np.float32)

    def reset(self):
        super().reset()
        self.ep = self.make_energyplus()
        self.ep.start()
        obs, over = self.ep.get_obs()
        # print(f"{obs}, {over}")
        keys, values = dataset.flatten_observation(obs)
        values = pandas.DataFrame({key: [val] for key, val in zip(keys, values)})
        values = preprocess_time(values)
        # print(values)
        return np.array(values)[0], {}

    def step(self, action):
        # Do something about the actions
        a = self.action_transform(action)
        obs, finished = self.ep.step(a)
        # print(f"{obs}, {finished}")
        if not finished:
            keys, values = dataset.flatten_observation(obs)
            values = pandas.DataFrame({key: [val] for key, val in zip(keys, values)})
            values = preprocess_time(values)
            values = np.array(values)[0]
            self.last_obs = values
            return (values, self.reward_fn(obs), False, False, {})
        else:
            return (self.last_obs, 0.0, True, False, {})
