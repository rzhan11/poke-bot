import numpy as np
from gymnasium.spaces import Box, Space

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    RandomPlayer,
)

from poke_env.ps_client.account_configuration import (
    AccountConfiguration,
    CONFIGURATION_FROM_PLAYER_COUNTER,
)
from . import embed

    
from ray.runtime_context import get_runtime_context
def worker_id_fn():
    worker_id = get_runtime_context().get_worker_id()
    worker_id = worker_id[:6]
    return worker_id

def create_worker_account_config(base_name, worker_id):
    raw_name = f"{base_name}-{worker_id}"
    CONFIGURATION_FROM_PLAYER_COUNTER.update([raw_name])
    username = f"{raw_name} {CONFIGURATION_FROM_PLAYER_COUNTER[raw_name]}"
    if len(username) > 18:
        assert False, "username too long"
    return AccountConfiguration(
        username=username,
        password=None,
    )

def create_player_fn(*, player_class, base_name, **kwargs):
    def create_player(worker_id):
        return player_class(
            account_configuration=create_worker_account_config(base_name, worker_id),
            **kwargs
        )
    return create_player

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, config_dict):
        base_config = config_dict["base_config"]
        custom_config = config_dict["custom_config"]
        ray_config = config_dict["ray_config"]

        self._init(**custom_config)
        self._init_ray(**ray_config, base_config=base_config)
        super().__init__(**base_config)

    def _init_ray(self, *, opponent_fn, base_name, base_config):
        # get worker id
        worker_id = worker_id_fn()

        ## update opponent
        base_config["opponent"] = opponent_fn(worker_id) if callable(opponent_fn) else opponent_fn 

        # we may pass in a function that returns the account (since the username may be worker-specific)
        base_config["account_configuration"] = create_worker_account_config(base_name, worker_id)

    def _init(self, input_len):
        self._input_len = input_len


    def calc_reward(self, last_battle, current_battle) -> float:
        reward = self.custom_reward(
            current_battle, 
            fainted_value=5.0,
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

        # if battle.won == True or battle.lost == True:
        #     current_value = victory_score
        # else:
        #     current_value = my_score - opp_score + victory_score
        
        current_value += victory_score

        ## END OF RZ MODS
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return

    def embed_battle(self, battle: AbstractBattle):
        emb = embed.BattleEmbed(battle)

        emb_dict = emb.embed_dict()
        emb_arr = embed.convert_embed_dict_to_ndarray(emb_dict)

        return emb_arr

    def describe_embedding(self) -> Space:
        return Box(
            low=-100,
            high=1000,
            shape=(self._input_len,),
            dtype=np.float32,
        )