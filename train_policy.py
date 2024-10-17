import math
import time
import random
import os

random.seed(time.time())
import typing
from typing import List
from pathlib import Path

from d3rlpy.algos import SAC
from d3rlpy.algos import DQNConfig, TD3Config, SACConfig, BCConfig, IQLConfig, SAC
import d3rlpy.dataset as d3d

import d3rlpy
import numpy as np
import pandas
import pickle

from dataset import RunResult
import dataset

import myeplus
import policy
import lzma
import typer
import wandb
import myeplusenv
import utils

dataset_files = [
    "pickles/day_night_1.pkl.xz",
    "pickles/random_walk_1.pkl.xz",
    "pickles/random_walk_2.pkl.xz",
]

app = typer.Typer()

GAMMA = 0.8
EVAL_TIME_STEP = 5000


def make_eval_env():
    return myeplus.EnergyPlus(
        *dataset.config,
        instance="eval",
        max_steps=EVAL_TIME_STEP,
    )


def load(filename: str):
    return pickle.load(lzma.open(filename, "rb"))


def concat_runresults(runresults: list[RunResult]):
    observations = np.concatenate(
        [np.array(utils.preprocess_time(rr.observations)) for rr in runresults], axis=0
    )
    actions = np.concatenate([np.array(rr.actions) for rr in runresults], axis=0)
    # preprocess_actions(df.actions),
    rewards = np.concatenate([np.array(rr.rewards) for rr in runresults], axis=0)
    terminals = np.concatenate(
        [
            np.concatenate([np.zeros(len(rr.rewards) - 1), np.ones(1)])
            for rr in runresults
        ],
        axis=0,
    )
    return (observations, actions, rewards, terminals)


@app.command()
def train_offline_sac(
    files: List[Path],
    name: str = "",
    gamma: float = GAMMA,
    n_steps: int = 200_000,
    n_steps_per_epoch: int = 5000,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
):
    runresults = list(map(load, files))

    obs, act, rew, term = concat_runresults(runresults)
    ds = d3d.MDPDataset(
        obs,
        act,
        rew,
        term,
        action_space=d3rlpy.constants.ActionSpace.CONTINUOUS,
    )

    eval_env = myeplusenv.EnergyPlusEnv(
        make_eval_env,
        dataset.compute_reward_fn,
    )

    model = SACConfig(
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        observation_scaler=d3rlpy.preprocessing.MinMaxObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    ).create()

    logger_adapter = d3rlpy.logging.CombineAdapterFactory(
        [
            d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
            d3rlpy.logging.WanDBAdapterFactory(),
        ]
    )
    # model = DQNConfig().create(device=None)
    # for _ in range(100):
    #     model.fit(ds, 100)

    model.fit(
        ds,
        n_steps,
        n_steps_per_epoch,
        logger_adapter=logger_adapter,
        experiment_name=f"offline_sac{name}",
        evaluators={"evaluation": d3rlpy.metrics.EnvironmentEvaluator(eval_env)},
    )


@app.command()
def train_offline_iql(
    files: List[Path],
    name: str = "",
    gamma: float = GAMMA,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    n_steps: int = 200_000,
    n_steps_per_epoch: int = 5000,
):
    runresults = list(map(load, files))

    obs, act, rew, term = concat_runresults(runresults)
    ds = d3d.MDPDataset(
        obs,
        act,
        rew,
        term,
        action_space=d3rlpy.constants.ActionSpace.CONTINUOUS,
    )

    eval_env = myeplusenv.EnergyPlusEnv(
        make_eval_env,
        dataset.compute_reward_fn,
    )

    model = IQLConfig(
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        observation_scaler=d3rlpy.preprocessing.MinMaxObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    ).create()

    logger_adapter = d3rlpy.logging.CombineAdapterFactory(
        [
            d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
            d3rlpy.logging.WanDBAdapterFactory(),
        ]
    )

    model.fit(
        ds,
        n_steps,
        n_steps_per_epoch,
        logger_adapter=logger_adapter,
        experiment_name=f"offline_iql{name}",
        evaluators={"evaluation": d3rlpy.metrics.EnvironmentEvaluator(eval_env)},
    )


@app.command()
def train_online_sac(
    start_epsilon: float = 0.5,
    end_epsilon: float = 0.1,
    n_steps: int = 200_000,
    n_steps_per_epoch: int = 5000,
    gamma: float = GAMMA,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    replay_buffer_size: int = 10000,
):
    def unshape_action(v):
        return

    def make_train_env():
        return myeplus.EnergyPlus(
            *dataset.config, instance="train", max_steps=EVAL_TIME_STEP
        )

    # used to calculate the mean and std reward.
    runresults: list[RunResult] = [load(dataset_file) for dataset_file in dataset_files]

    env = myeplusenv.EnergyPlusEnv(
        make_train_env,
        dataset.compute_reward_fn,
    )
    eval_env = myeplusenv.EnergyPlusEnv(
        make_eval_env,
        dataset.compute_reward_fn,
    )
    fifo = d3d.FIFOBuffer(limit=replay_buffer_size)

    buf = d3d.ReplayBuffer(
        fifo,
        d3d.BasicTransitionPicker(),
        d3d.BasicTrajectorySlicer(),
        d3d.BasicWriterPreprocess(),
        env=env,
    )

    # setup explorers
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=end_epsilon,
        duration=n_steps,
    )

    obs, act, rew, term = concat_runresults(runresults)
    # feu TD3Config
    model = SACConfig(
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        observation_scaler=d3rlpy.preprocessing.MinMaxObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(
            mean=np.mean(rew),
            std=np.std(rew),
        ),
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    ).create()

    logger_adapter = d3rlpy.logging.CombineAdapterFactory(
        [
            d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
            d3rlpy.logging.WanDBAdapterFactory(),
        ]
    )

    model.fit_online(
        env,
        buf,
        eval_env=eval_env,
        explorer=explorer,
        n_steps=n_steps,  # the number of total steps to train.
        n_steps_per_epoch=n_steps_per_epoch,
        logger_adapter=logger_adapter,
        experiment_name="online_sac",
    )
    return model


def randlog(start, end):
    return np.exp(random.random() * (np.log(end) - np.log(start)) + np.log(start))


@app.command()
def offline_sac_random_params():
    gamma = 1 - randlog(0.01, 0.1)
    actor_learning_rate = randlog(3e-5, 1e-3)
    critic_learning_rate = randlog(3e-5, 1e-3)

    train_offline_sac(
        dataset_files,
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        n_steps=2_000_000,
    )


@app.command()
def offline_iql_random_params():
    gamma = 1 - randlog(0.01, 0.1)
    actor_learning_rate = randlog(3e-5, 1e-3)
    critic_learning_rate = randlog(3e-5, 1e-3)

    train_offline_iql(
        dataset_files,
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        n_steps=2_000_000,
    )


@app.command()
def online_random_params():
    gamma = 1 - randlog(0.01, 0.1)
    actor_learning_rate = randlog(3e-5, 1e-3)
    critic_learning_rate = randlog(3e-5, 1e-3)

    train_online_sac(
        gamma=gamma,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        n_steps=2_000_000,
    )


def test_something():
    mod = SAC.from_json("./params.json")
    mod.load_model("./model_200000.d3")
    return mod


@app.command()
def view_policy(f: Path):
    algo = d3rlpy.load_learnable(f)

    eval_env = myeplusenv.EnergyPlusEnv(
        make_eval_env,
        dataset.compute_reward_fn,
    )

    truncated = False
    stopped = False
    obs, info = eval_env.reset()
    while not (truncated or stopped):
        true_obs = info["raw_observation"]
        act = algo.predict(obs.reshape(1, -1))
        print(true_obs)
        print(act[0, :])
        obs, r, truncated, stopped, info = eval_env.step(act[0, :])
        print(r)


if __name__ == "__main__":
    app()
