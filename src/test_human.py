import asyncio
from max_damage_player import MaxDamagePlayer
from collections import namedtuple
from poke_env.ps_client import AccountConfiguration


async def test():
    num_battles = 100
    num_workers = 10

    player = MaxDamagePlayer(
        account_configuration=AccountConfiguration(username="RZMaxDMG", password=None),
        battle_format="gen8randombattle", 
        max_concurrent_battles=num_workers,
        save_replays=True,
    )

    player = MaxDamagePlayer(
        account_configuration=AccountConfiguration(username="RZMaxDMG", password=None),
        battle_format="gen8randombattle", 
        max_concurrent_battles=num_workers,
        save_replays=True,
    )


    await player.accept_challenges(None, n_challenges=1, packed_team=None)


asyncio.run(test())