import asyncio
import time
import json

from poke_env.player import Player, RandomPlayer, BattleOrder
from poke_env.environment import Battle, Move

def get_move_dict(move):
    return {attr: getattr(move, attr) for attr in dir(move) 
            if not attr.startswith('_') and not callable(getattr(move, attr))}


class MaxDamagePlayer(Player):
    def choose_move(self, battle: Battle) -> BattleOrder:
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move: Move = max(battle.available_moves, key=lambda move: move.base_power)
            # for move in battle.available_moves:
            #   print("Move", move)
            #   move_dict = get_move_dict(move)
            #   with open(f"moves/{move.id}.txt", "w") as f:
            #       f.write(json.dumps({str(k): str(v) for k, v in move_dict.items()}, indent=2))

            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    num_workers = 1
    num_battles = 100

    # We create two players.
    random_player = RandomPlayer(battle_format="gen8randombattle", max_concurrent_battles=num_workers)
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle", max_concurrent_battles=num_workers)

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=num_battles)

    print(f"Max damage player won {max_damage_player.n_won_battles} / {num_battles} battles [this took {time.time() - start} seconds]")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())