import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.common import get_ids
import profile

idffile = profile.PROFILE + "/US+MF+CZ1AWH+elecres+crawlspace+IECC_2006.idf"
newenv = sinergym.envs.EplusEnv(idffile,  profile.weatherfile("honolulu.epw"))
