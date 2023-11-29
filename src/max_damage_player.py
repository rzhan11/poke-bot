import asyncio
import time
import json

from typing import List, Dict
from poke_env.player import Player, RandomPlayer, BattleOrder
from poke_env.environment import Battle, Move, Pokemon

import sys
sys.path.append("../python")
from rzlib.env import embed

def get_move_dict(move):
    return {attr: getattr(move, attr) for attr in dir(move) 
            if not attr.startswith('_') and not callable(getattr(move, attr))}

def print_pok(pok: Pokemon, tabs=0):
    tabstr = "\t" * tabs
    print(tabstr, pok, "hp", round(pok.current_hp_fraction, 3), "status", pok.status)
    print(tabstr, "item", pok.item, ", ability", pok.ability)
    print(tabstr, "moves")
    print_moves(pok.moves.values(), tabs=tabs+1)


def print_team(team: Dict[str, Pokemon], tabs=0):
    for pok in team.values():
        print_pok(pok, tabs=tabs)

def print_move(move: Move, tabs=0):
    print("\t" * tabs, move, "base", move.base_power, "acc", move.accuracy, "dmg", move.damage, "pp", move.current_pp)

def print_moves(moves: List[Move], tabs=0):
    for move in moves:
        print_move(move, tabs=tabs)

def report_info(battle: Battle):
    print("battle tag", battle.battle_tag, "turn", battle.turn)
    print("My active:")
    print_pok(battle.active_pokemon, tabs=1)
    print("Opp active:")
    print_pok(battle.opponent_active_pokemon, tabs=1)
    print("My team")
    print_team(battle.team, tabs=1)
    print("Opp team")
    print_team(battle.opponent_team, tabs=1)
    print()



class MaxDamagePlayer(Player):
    def choose_move(self, battle: Battle) -> BattleOrder:
        # report_info(battle)
        emb = embed.BattleEmbed(battle)
        res = emb.embed_dict(with_tags=False)
        # # res = embed.remove_unknown_dicts(res)
            
        # with open("test_dict.txt", "w") as f:
        #     json.dump(res, f, indent=2)
            
        with open("test_dict_tag.txt", "w") as f:
            json.dump(emb.embed_dict(with_tags=True), f, indent=2)

        arr = list(embed.convert_embed_dict_to_ndarray(res))
        # print(min(arr), max(arr))
        print(len(arr), type(embed.convert_embed_dict_to_ndarray(res)))
        # with open("test_arr.txt", "w") as f:
        #     json.dump(list(arr), f, indent=2)

        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move: Move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    num_workers = 2
    num_battles = 1000

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle", 
        max_concurrent_battles=num_workers,
        save_replays=True,
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8randombattle", 
        max_concurrent_battles=num_workers,
        save_replays=True,
    )

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=num_battles)

    print(f"Max damage player won {max_damage_player.n_won_battles} / {num_battles} battles [this took {time.time() - start} seconds]")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())