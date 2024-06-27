import jax.numpy as np
import jax.typing as jtyping
from dataclasses import dataclass
import typing


@dataclass
class Trajectory:
    _start_state: list
    _action_probability: list
    _action: list
    _reward: list
    _end_state: list

    def __init__(self):
        self._start_state = []
        self._action_probability = []
        self._action = []
        self._reward = []
        self._end_state = []

    def add_transition(self, s1, p, a, r, s2):
        self._start_state.append(s1)
        self._action_probability.append(p)
        self._action.append(a)
        self._reward.append(r)
        self._end_state.append(s2)

    def get_arrays(self):
        n = len(self._start_state)
        return (
            np.reshape(np.array(self._start_state), (n, -1)),
            np.reshape(np.array(self._action_probability), (n, -1)),
            np.reshape(np.array(self._action), (n, -1)),
            np.reshape(np.array(self._reward), (n, -1)),
            np.reshape(np.array(self._end_state), (n, -1)),
        )

    def sum_of_rewards(self):
        return sum(self._reward)

    def len(self):
        return len(self._action)

    def rewards_to_go(self, gamma):
        out = []
        out.append(self._reward[-1])
        # traverse the rewards from the second last to the first
        for r in self._reward[::-1][1:]:
            out.append(r + gamma * out[-1])
        out.reverse()
        return np.array(out)


def merge_trajectories(
    trajs: list[Trajectory], gamma
) -> typing.Tuple[
    jtyping.ArrayLike,
    jtyping.ArrayLike,
    jtyping.ArrayLike,
    jtyping.ArrayLike,
    jtyping.ArrayLike,
]:
    start_state = []
    action_probability = []
    action = []
    reward_to_go = []
    end_state = []
    for traj in trajs:
        s1, p, a, _, s2 = traj.get_arrays()
        r = np.reshape(traj.rewards_to_go(gamma), (traj.len(), -1))
        start_state += [s1]
        action_probability += [p]
        action += [a]
        reward_to_go += [r]
        end_state += [s2]

    return (
        np.vstack(start_state),
        np.vstack(action_probability),
        np.vstack(action),
        np.vstack(reward_to_go),
        np.vstack(end_state),
    )


def mean_reward(trajectories):
    return sum(traj.sum_of_rewards() for traj in trajectories) / len(trajectories)
