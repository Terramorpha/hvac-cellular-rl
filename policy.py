import random
import typing

Observation = typing.Any
Action = typing.Any
Policy = typing.Callable[[Observation], Action]


def combine(pol1: Policy, pol2: Policy) -> Policy:
    """take two policies and create a new policy combining both of their
    behaviors."""

    def policy_sum(obs):
        return {
            **pol1(obs),
            **pol2(obs),
        }

    return policy_sum


def const(**kwargs) -> Policy:
    """Make a poliy that always outputs `value` for the given setpint name."""

    def const_policy(_):
        return {setpoint: value for setpoint, value in kwargs.items()}

    return const_policy


def day_night(day_policy: Policy, night_policy: Policy) -> Policy:
    def day_night_policy(obs):
        if 7.0 <= obs["time"]["time_of_day"].inner <= 19:
            return day_policy(obs)
        else:
            return night_policy(obs)

    return day_night_policy


def random_walk(actuator: str, low: float, high: float, sigma=1) -> Policy:
    def random_walk_policy(obs):
        old_val = obs["actuators"][actuator].inner
        step = random.normalvariate(0, sigma)
        new_val = old_val + step
        new_val = max(low, min(new_val, high))
        return {actuator: new_val}

    return random_walk_policy


def chain(f: Policy, *modifications) -> Policy:
    def g(obs: Observation) -> Action:
        act = f(obs)
        for m in modifications:
            act = m(act)
        return act

    return g


def force_actuator_smaller(action, actuator1, actuator2):
    """Modify an action to force the value of `actuator1` to always be smaller
    than the value of `actuator2`.

    """

    action = {**action}
    action[actuator1] = min(action[actuator1], action[actuator2])
    return action
