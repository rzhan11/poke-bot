import asyncio

import numpy as np
import json

from poke_env.player import Player, RandomPlayer

import sys
sys.path.append("/Users/richardzhan/cs/15888/poke/python")
from rzlib.env import embed
from rzlib.env.simple_rl_player import SimpleRLPlayer

ref_rl_player = SimpleRLPlayer(None, None)

class CustomRandomPlayer(Player):
    def choose_move(self, battle):
        emb = embed.BattleEmbed(battle)
        res = emb.embed_dict(with_tags=True)
        res_no_tag = emb.embed_dict(with_tags=False)
        res_arr = embed.convert_embed_dict_to_ndarray(res_no_tag)
        # print(embed.embed_dumps(res))
        # print(len(res_arr))

        # reward = ref_rl_player.calc_reward(None, battle)

        return self.choose_random_move(battle)



_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"
def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())

import logging

async def main():
    print("ou_embed_tester()")
    team_1 = load_team(_team1_fname)
    team_2 = load_team(_team2_fname)

    # We create two players.
    opponent = RandomPlayer(
        battle_format="gen8ou", 
        team=team_1, 
        max_concurrent_battles=10
    )

    my_player = CustomRandomPlayer(
        battle_format="gen8ou", 
        team=team_2, 
        max_concurrent_battles=10,
        # log_level=logging.DEBUG,
    )

    # Now, let's evaluate our player
    await my_player.battle_against(opponent, n_battles=1)

    print(f"{my_player.username} player won {my_player.n_won_battles} / {my_player.n_finished_battles} battles")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())