import asyncio

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
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
    def __init__(self, config_dict):
        super().__init__(**config_dict)

        self._data: GenData = GenData.from_gen(8)

    def calc_reward(self, last_battle, current_battle) -> float:
        reward = self.custom_reward(
            current_battle, 
            fainted_value=10.0, 
            # hp_value=5.0, 
            starting_value=0.0, 
            # status_value=2.5, 
            victory_value=100.0, 
        )
        return reward
    
    def custom_reward(self, battle, *, 
            fainted_value=0.0, 
            hp_value=0.0,
            starting_value=0.0,
            status_value=0.0,
            victory_value=0.0,
            ):

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        ## RZ MODS BELOW
    
        def count_stats(team):
            total_hp = 0.0
            num_faint = 0
            num_status = 0
            for mon in team.values():
                total_hp += mon.current_hp_fraction
                num_faint += int(mon.fainted)
                num_status += int(mon.status is not None)
            
            return {"hp": total_hp, "faint": num_faint, "status": num_status, "num_team": len(team)}
        
        def score_stats(stats):
            return (stats["hp"] - stats["num_team"]) * hp_value - stats["faint"] * fainted_value - stats["status"] * status_value
        
        my_stats = count_stats(battle.team)
        opp_stats = count_stats(battle.opponent_team)

        my_score = score_stats(my_stats)
        opp_score = score_stats(opp_stats)

        victory_score = (int(battle.won == True) - int(battle.lost == True)) * victory_value

        if battle.won == True or battle.lost == True:
            current_value = victory_score
        else:
            current_value = my_score - opp_score + victory_score

        ## END OF RZ MODS
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return

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