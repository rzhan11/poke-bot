import numpy as np

from ray.rllib.policy.policy import Policy

from poke_env.player import (
    Player,
)
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from .simple_rl_player import embed_battle_internal




class PolicyPlayer(Player):
    _policy: Policy
    def __init__(self, policy: Policy, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._policy = policy

    def choose_move(self, battle: AbstractBattle):

        obs = embed_battle_internal(battle)
        action, rnn_state, extra_features = self._policy.compute_single_action(obs)

        print("Action", action, extra_features)
        # print("Action", action, rnn_state, extra_features)

        move = self.action_to_move(action, battle)
        print("My move", move)
        return move


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
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            print(f"Choosing random move, received action:{action}")
            return self.choose_random_move(battle)