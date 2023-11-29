import asyncio
from ou_embed_tester import CustomRandomPlayer
from collections import namedtuple
from poke_env.ps_client import AccountConfiguration


_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())
    
team_1 = load_team(_team1_fname)
team_2 = load_team(_team2_fname)

async def test():
    player = CustomRandomPlayer(
        account_configuration=AccountConfiguration(username="RZMaxDMG", password=None),
        battle_format="gen8ou", 
        max_concurrent_battles=1,
        save_replays=True,
        # battle_format="gen8randombattle", 
        team=team_1,
    )


    await player.accept_challenges(None, n_challenges=1, packed_team=None)


asyncio.run(test())