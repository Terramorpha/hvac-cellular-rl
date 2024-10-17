import myeplus
import utils


def _penality(mintmp, t, maxtmp):
    return min(t - mintmp, 0) - max(t - maxtmp, 0)


def combine(a, b):
    def combined_reward(obs):
        return a(obs) + b(obs)

    return combined_reward


TEMPERATURE_ZONE_BLACKLIST = [
    "environment",
    "attic",
    "Breezeway",
    "crawlspace",
]


def remove_blacklisted(obs_tmp):
    return {k: v for k, v in obs_tmp.items() if k not in TEMPERATURE_ZONE_BLACKLIST}


def range_reward(mintmp, maxtmp):
    def range_reward_fn(obs):
        temperatures = utils.collect(
            remove_blacklisted(obs["temperature"]), myeplus.Variable
        )
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
