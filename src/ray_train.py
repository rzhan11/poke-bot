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

ppo_config = (
    PPOConfig()
    .framework("torch")
    .training(
        model={"fcnet_hiddens": [1024, 1024, 1024]},
        train_batch_size=10*1024,
        sgd_minibatch_size=2048,
        gamma=0.99,
        lr=0.001,
        use_critic=True,
        use_gae=True,
        shuffle_sequences=True,

        # kl_coeff=0.3,
        # target_network_update_freq=500,
        # gamma=0.99,
        # lr=0.001,
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
        # num_cpus=_num_cpus,
        # num_gpus=_num_gpus,
    )
    .environment(env=select_env, env_config={
        "base_config": {
            "battle_format": _battle_format,
            "start_challenging": True,
            # "team": load_team(_team2_fname), # my team
        },
        "ray_config": {
            "opponent_fn": create_player_fn(
                player_class=MaxBasePowerPlayer,
                base_name="Max",
                battle_format=_battle_format, 
                # team=load_team(_team1_fname),
            ),
            "base_name": "PPO",
        },
        "custom_config": {
            "input_len": 514,
        },
    })
    .fault_tolerance(
        recreate_failed_workers=True,
        max_num_worker_restarts=100,
        delay_between_worker_restarts_s=10,
        num_consecutive_worker_failures_tolerance=3,
    )
)

_num_iters = 10000
_checkpoint_freq = 10
_checkpoint_base_folder_name = "../results/ppo"
_checkpoint_folder = get_checkpoint_folder_version(_checkpoint_base_folder_name)

# _use_checkpoint = "../results/ppo_1/"
_use_checkpoint = None
_checkpoint_iter = 350

if _use_checkpoint is not None: ## Load from a checkpoint
    from ray.rllib.algorithms.algorithm import Algorithm

    assert type(_use_checkpoint) is str, f"_use_checkpoint should be a str, received {type(_use_checkpoint)}"
    _checkpoint_folder = Path(_use_checkpoint)
    algo = Algorithm.from_checkpoint(
        create_save_folder(
            base=_checkpoint_folder, 
            cur_iter=_checkpoint_iter,
        )
    )

    print("Using checkpoint", _use_checkpoint)
    start_iter = _checkpoint_iter + 1
else:
    algo = ppo_config.build()
    start_iter = 0

print(json.dumps({
    "_num_cpus": _num_cpus,
    "_num_gpus": _num_gpus,
    "_num_rollout_workers": _num_rollout_workers,
}, indent=2))


orig_start_time = time.time()
_checkpoint_folder.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint folder {_checkpoint_folder}")
for cur_iter in range(start_iter, _num_iters):
    print("\n\nIter", cur_iter)

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

    # checkpoint the algo (every X intervals + on the last iteration)
    if cur_iter % _checkpoint_freq == 0 or cur_iter == _num_iters - 1:
        save_folder = create_save_folder(_checkpoint_folder, cur_iter)
        # print(f"Checkpointing to {save_path}")
        save_res = algo.save(checkpoint_dir=save_folder)
        print("Saved", save_res.checkpoint.path)
    # assert False

    # eval_start_time = time.time()
    # eval_res = algo.evaluate()
    # print("eval_reward", eval_res["evaluation"]["episode_reward_mean"])
    # print(f"Eval time: {(time.time() - eval_start_time):.0f} s")

    # print(f"Total time: {(time.time() - orig_start_time):.0f} s")
