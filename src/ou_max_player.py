import asyncio

import numpy as np

from poke_env.player import Player, RandomPlayer


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def teampreview(self, battle):
        mon_performance = {}

        # For each of our pokemons
        for i, mon in enumerate(battle.team.values()):
            # We store their average performance against the opponent team
            mon_performance[i] = np.mean(
                [
                    teampreview_performance(mon, opp)
                    for opp in battle.opponent_team.values()
                ]
            )

        # We sort our mons by performance
        ordered_mons = sorted(mon_performance, key=lambda k: -mon_performance[k])

        # We start with the one we consider best overall
        # We use i + 1 as python indexes start from 0
        #  but showdown's indexes start from 1
        return "/team " + "".join([str(i + 1) for i in ordered_mons])


def teampreview_performance(mon_a, mon_b):
    # We evaluate the performance on mon_a against mon_b as its type advantage
    a_on_b = b_on_a = -np.inf
    for type_ in mon_a.types:
        if type_:
            a_on_b = a_on_b
    # We do the same for mon_b over mon_a
    for type_ in mon_b.types:
        if type_:
            b_on_a = b_on_a
    # Our performance metric is the different between the two
    return a_on_b - b_on_a


async def main():
    team_1 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish
"""
    team_2 = """
Togekiss @ Leftovers
Ability: Serene Grace
EVs: 248 HP / 8 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Air Slash
- Nasty Plot
- Substitute
- Thunder Wave

Galvantula @ Focus Sash
Ability: Compound Eyes
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Sticky Web
- Thunder Wave
- Thunder
- Energy Ball
"""

    # We create two players.
    opponent = MaxDamagePlayer(
        battle_format="gen8ou", team=team_1, max_concurrent_battles=10
    )

    max_damage_player = MaxDamagePlayer(
        battle_format="gen8ou", team=team_2, max_concurrent_battles=10
    )

    # Now, let's evaluate our player
    await max_damage_player.battle_against(opponent, n_battles=100)

    print("Max damage player won %d / 100 battles" % max_damage_player.n_won_battles)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())