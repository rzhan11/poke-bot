import asyncio

import numpy as np
from gym.spaces import Box, Space
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
    wrap_for_old_gym_api,
)
from poke_env.data import GenData

from . import embed

_input_len = 514

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: GenData = GenData.from_gen(8)

    def calc_reward(self, last_battle, current_battle) -> float:
        reward = self.reward_computing_helper(
            current_battle, 
            fainted_value=10.0, 
            # hp_value=5.0, 
            # number_of_pokemons=2, # how many pokemon we start with 
            # starting_value=0.0, 
            # status_value=2.5, 
            victory_value=100.0, 
        )
        return reward

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        emb = embed.BattleEmbed(battle)

        emb_dict = emb.embed_dict()
        emb_arr = embed.convert_embed_dict_to_ndarray(emb_dict)

        return emb_arr

    def describe_embedding(self) -> Space:
        return Box(
            low=-100,
            high=1000,
            shape=(_input_len,),
            dtype=np.float32,
        )