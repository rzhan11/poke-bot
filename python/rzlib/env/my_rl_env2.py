import asyncio

from typing import Union, List, Dict

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

from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    Player,
    BaseRLPokeEnv, ## added
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
)
from poke_env.data import GenData

from . import embed

_input_len = 514

class MyRLEnv(BaseRLPokeEnv):
    def __init__(self, config_dict):
        super().__init__(**config_dict)

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

    def action_space_size(self) -> int:
        return len(self._ACTION_SPACE)
    


    _ACTION_SPACE = list(range(4 * 4 + 6))
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.agent.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.agent.create_order(battle.available_switches[action - 16])
        else:
            return self.agent.choose_random_move(battle)