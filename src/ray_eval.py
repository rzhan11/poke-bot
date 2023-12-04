# standard imports
import asyncio
import numpy as np
import json
import time
from pathlib import Path
import os

## gym imports
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env

## poke-env imports
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)

## import RZLib
import sys 
sys.path.append("/Users/richardzhan/cs/15888/poke/python")

from rzlib.env.simple_rl_player import (
    SimpleRLPlayer,
    create_player_fn,
)

import pickle

## ray imports
import ray
_num_cpus = 64
_num_gpus = 4
ray.init(
    # local_mode=True,
    num_cpus=_num_cpus,
    num_gpus=_num_gpus,
    include_dashboard=True,
    dashboard_host="127.0.0.1",
    dashboard_port=6100,
    ignore_reinit_error=True,
)

## register the environment
from ray.tune.registry import register_env
select_env = "simple-rl-player-poke"
register_env(select_env, lambda config: SimpleRLPlayer(config))

## create PPO algo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm


# _battle_format = "gen8ou"
_battle_format = "gen8randombattle"
_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())

def get_checkpoint_folder_version(folder_name):
    i = 0
    while True:
        path = Path(f"{folder_name}_{i}")
        if path.exists():
            i += 1
        else:
            break
    return path

def create_save_folder(base, cur_iter):
    return base / f"iter_{cur_iter:05d}/"

# num_eval_workers: https://discuss.ray.io/t/num-gpu-rollout-workers-learner-workers-evaluation-workers-purpose-resource-allocation/10159

_num_rollout_workers = int(1 * _num_cpus)
_num_eval_workers = 0 ## This is probably not useful... (i think? see above link)
_num_envs_per_worker = 1
_num_workers = _num_rollout_workers + _num_eval_workers

_max_iter = 2300
_checkpoint_freq = 10
_checkpoint_folder = Path("../results/ppo_day1_eval")

print(f"Checkpoint folder {_checkpoint_folder}")
orig_start_time = time.time()
for cur_iter in range(0, _max_iter + 1, _checkpoint_freq):
    save_folder = create_save_folder(base=_checkpoint_folder, cur_iter=cur_iter)
    print("Loading checkpoint", cur_iter, save_folder)
    

    config_path = save_folder / "algorithm_state.pkl"
    with open(config_path, "rb") as f:
        config: PPOConfig = pickle.load(f)["config"].copy(copy_frozen=False)

    print("eval_workers", config.evaluation_num_workers)
    print("eval_interval", config.evaluation_interval)
    
    config = config.rollouts(
        num_rollout_workers=16,
        num_envs_per_worker=1,
    ).evaluation(
        evaluation_num_workers=16,
        evaluation_interval=1,
    )

    print("config", type(config))
    print("eval_workers", config.evaluation_num_workers)
    print("eval_interval", config.evaluation_interval)
    # print("eval_interval", config.num_rollout_workers)
    # config['evaluation_num_workers'] = 3
    # config['evaluation_interval'] = 1  # <-- HERE: must set this to > 0!
    # config['num_workers'] = 0
    
    
    algo = config.build()
    algo.restore(str(save_folder))
    # algo: Algorithm = Algorithm.from_checkpoint(
    #     save_folder
    # )
    # # algo.config.evaluation(evaluation_num_workers=_num_eval_workers).rollouts(num_rollout_workers=32)
    # # algo.evaluation_num_workers
    # import time
    # time.sleep(100)

    # state_file = 
    if cur_iter == 0:
        print("Config")
        print(json.dumps(algo.config.to_dict(), indent=2, default=lambda o: f"<not serializable: {str(o)}>"))
    
    eval_res = algo.evaluate()

    algo.stop()
    break
    
    # print("eval_reward", eval_res["evaluation"]["episode_reward_mean"])

    # print(f"Total time: {(time.time() - orig_start_time):.0f} s")

    # break