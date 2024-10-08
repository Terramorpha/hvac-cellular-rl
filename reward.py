import myeplus
import utils


def _penality(mintmp, t, maxtmp):
    return max(t - maxtmp, 0) - min(t - mintmp, 0)


def combine(a, b):
    def combined_reward(obs):
        return a(obs) + b(obs)

    return combined_reward


def range_reward(mintmp, maxtmp):
    def range_reward_fn(obs):
        temperatures = utils.collect(obs["temperature"], myeplus.Variable)
        return -sum(map(lambda t: _penality(mintmp, t, maxtmp), temperatures)) / len(
            temperatures
        )

    return range_reward_fn


def day_night_reward(day, night):
    def day_night_reward_fn(obs):
        if 7 <= obs["time"]["time_of_day"].inner <= 19:
            return day(obs)
        else:
            return night(obs)

    return day_night_reward_fn


def energy_reward():
    def energy_reward_fn(obs):
        temperatures = utils.collect(obs["energy"], myeplus.Variable)
        return -sum(temperatures)

    return energy_reward_fn
