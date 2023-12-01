import asyncio

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    RandomPlayer,
)
from poke_env.data import GenData

import sys 
sys.path.append("/Users/richardzhan/cs/15888/poke/python")

from rzlib.env import embed
from rzlib.env.my_rl_env import MyRLEnv

## ray imports
import ray
ray.init(
    ignore_reinit_error=True,
    local_mode=True,
)

## register the environment
from ray.tune.registry import register_env
select_env = "simple-rl-player-poke"
register_env(select_env, lambda config: MyRLEnv(config))


## create PPO algo
from ray.rllib.algorithms.ppo import PPOConfig

_battle_format = "gen8ou"
_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())
    
def create_random_player():
    return RandomPlayer(battle_format=_battle_format, team=load_team(_team1_fname))

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=select_env, env_config={
        "battle_format": _battle_format,
        "opponent": create_random_player(),
        "start_challenging": True,
        "team": load_team(_team2_fname),
    })
    .build(use_copy=False)
)

# for i in range(10):
#     result = algo.train()
#     print(result)

#     if i % 5 == 0:
#         checkpoint_dir = algo.save().checkpoint.path
#         print(f"Checkpoint saved in directory {checkpoint_dir}")



# _battle_format = "gen8ou"
# _team1_fname = "../data/team1.txt"
# _team2_fname = "../data/team2.txt"

# def load_team(fname):
#     # load_bot()
#     with open(fname, "r") as f:
#         return "".join(f.readlines())

# def create_random_player():
#     return RandomPlayer(battle_format=_battle_format, team=load_team(_team1_fname))

# def create_max_power_player():
#     return MaxBasePowerPlayer(battle_format=_battle_format, team=load_team(_team1_fname))

# def create_rl_player(opponent, **kwargs):
#     return SimpleRLPlayer(
#         battle_format=_battle_format,
#         opponent=opponent,
#         start_challenging=True,
#         team=load_team(_team2_fname),
#         **kwargs
#     )

