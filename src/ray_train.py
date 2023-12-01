# standard imports
import asyncio
import numpy as np
import json
import time

## gym imports
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env

## poke-env imports
from poke_env.ps_client.account_configuration import (
    AccountConfiguration,
    CONFIGURATION_FROM_PLAYER_COUNTER,
)
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    RandomPlayer,
)

## import RZLib
import sys 
sys.path.append("/Users/richardzhan/cs/15888/poke/python")

from rzlib.env.simple_rl_player import SimpleRLPlayer

## ray imports
import ray
_num_cpus = 8
_num_gpus = 0
ray.init(
    # local_mode=True,
    num_cpus=_num_cpus,
    num_gpus=_num_gpus,
    include_dashboard=True,
    dashboard_host="127.0.0.1",
    dashboard_port=8700,
    ignore_reinit_error=True,
)

## register the environment
from ray.tune.registry import register_env
select_env = "simple-rl-player-poke"
register_env(select_env, lambda config: SimpleRLPlayer(config))

## create PPO algo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig


_battle_format = "gen8ou"
_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())
    
from ray.runtime_context import get_runtime_context
def worker_id_fn():
    worker_id = get_runtime_context().get_worker_id()
    worker_id = worker_id[:6]
    return worker_id

def create_random_player(worker_id=0):
    return RandomPlayer(
        account_configuration=create_worker_account_config("Rand", worker_id),
        battle_format=_battle_format, 
        team=load_team(_team1_fname),
    )

def create_worker_account_config(base_name, worker_id):
    raw_name = f"{base_name}-{worker_id}"
    CONFIGURATION_FROM_PLAYER_COUNTER.update([raw_name])
    username = f"{raw_name} {CONFIGURATION_FROM_PLAYER_COUNTER[raw_name]}"
    if len(username) > 18:
        assert False, "username too long"
    return AccountConfiguration(
        username=username,
        password=None,
    )

_num_rollout_workers = 16
_num_eval_workers = 0
_num_envs_per_worker = 2
_num_workers = _num_rollout_workers + _num_eval_workers
algo = (
    DQNConfig()
    .framework("torch")
    .training(
        model={"fcnet_hiddens": [128, 128, 128]},
        target_network_update_freq=500,
        gamma=0.99,
        lr=0.001,
        # grad_clip=0.1,
        # optimizer=None,
        # train_batch_size=None,
    )
    .rollouts(
        num_rollout_workers=_num_rollout_workers,
        num_envs_per_worker=_num_envs_per_worker,
    )
    .evaluation(evaluation_num_workers=_num_eval_workers)
    .resources(
        num_cpus_per_worker=_num_cpus / _num_workers, 
        num_gpus_per_worker=_num_gpus / _num_workers,
    )
    .environment(env=select_env, env_config={
        "account_configuration": lambda worker_id: create_worker_account_config("RLP", worker_id),
        "worker_id_fn": worker_id_fn,
        "battle_format": _battle_format,
        "opponent_fn": lambda worker_id: create_random_player(worker_id),
        "start_challenging": True,
        "team": load_team(_team2_fname),
    })
    .build(use_copy=False)
)

orig_start_time = time.time()
for i in range(100):
    print("\n\nIter", i)

    train_start_time = time.time()
    train_res = algo.train()
    print("train", json.dumps({
        "training_iteration": train_res["training_iteration"],
        "episode_reward_mean": train_res["sampler_results"]["episode_reward_mean"],
        "episodes_this_iter": train_res["sampler_results"]["episodes_this_iter"],
        "episodes_total": train_res["episodes_total"],
        "timesteps_total": train_res["timesteps_total"],
        "counters": train_res["counters"],
        "timers": train_res["timers"],
        "time_this_iter_s": train_res["time_this_iter_s"],
        "time_total_s": train_res["time_total_s"],
    }, indent=2))
    print(f"Train time: {(time.time() - train_start_time):.0f} s")

    # assert False

    # eval_start_time = time.time()
    # eval_res = algo.evaluate()
    # print("eval_reward", eval_res["evaluation"]["episode_reward_mean"])
    # print(f"Eval time: {(time.time() - eval_start_time):.0f} s")

    # print(f"Total time: {(time.time() - orig_start_time):.0f} s")
