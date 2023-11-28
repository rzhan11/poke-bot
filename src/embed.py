from typing import List, Dict, Any

from poke_env.data import GenData
from poke_env.environment import Battle
from poke_env.environment import Pokemon
from poke_env.environment import Status
from poke_env.environment import Move, EmptyMove

import itertools
import numpy as np
from pathlib import Path
import json

## HYPERPARAMS
_cur_gen = 8
_static_data_folder = Path("../data")
_move_fpath = _static_data_folder / "move.json"
_item_fpath = _static_data_folder / "item.json"
_ability_fpath = _static_data_folder / "ability.json"


NUM_MOVES_PER_POKEMON = 4
NUM_POKEMON_PER_TEAM = 6

EMPTY_SPECIES = "empty_species"

GEN8_DATA = GenData.from_gen(_cur_gen)
with open(_move_fpath, "r") as f:
    MOVE_DICT = json.load(f)
with open(_item_fpath, "r") as f:
    ITEM_DICT = json.load(f)
with open(_ability_fpath, "r") as f:
    ABILITY_DICT = json.load(f)


def flatten_dict_gen(d, parent_key='', sep='_'):
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            yield from flatten_dict_gen(v, new_key, sep=sep)
        else:
            yield np.array([v])
            # yield (new_key, v)


def convert_embed_dict_to_ndarray(d: Dict):
    arr = list(flatten_dict_gen(d))
    arr = [np.array(a).ravel() for a in arr]
    return np.concatenate(arr)



def remove_unknown_dicts(d: Dict):
    if type(d) is not dict:
        return d
    
    if d.get("is_unknown", 0) == 1:
        return None
    if d.get("pok_data.is_unknown", 0) == 1:
        return None
    return {k: remove_unknown_dicts(v) for k, v in d.items()}

class AbstractEmbed():
    tags: Dict[str, Any]
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def embed(self) -> np.ndarray:
        data = self.embed_dict(recurse_dict=True)
        return data
        # return np.concatenate(data.values())
    
    def embed_dict(self, recurse_dict=True, with_tags=False) -> Dict:
        raise NotImplementedError

    def add_tags(self, data: Dict) -> Dict:
        assert len(set(data.keys()).intersection(set(self.tags))) == 0
        tags = {f"_{k}": v for k, v in self.tags.items()}
        data = {
            **data,
            **tags,
        }
        return data
    
class AbstractEndEmbed(AbstractEmbed):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def embed(self) -> np.ndarray:
        return np.array(self.embed_dict(with_tags=False).values())

    def embed_dict(self, with_tags=False, **kwargs) -> Dict:
        data = self.embed_raw_dict()
        if with_tags:
            data = self.add_tags(data)
        return data
    
    def embed_raw_dict(self) -> Dict:
        raise NotImplementedError
    
""" High-level embedding """
class AbstractHLEmbed(AbstractEmbed):
    features: Dict[str, "AbstractEmbed"]
    in_features: Dict[str, "AbstractEndEmbed"]
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def embed_dict(self, recurse_dict=True, with_tags=False):
        if recurse_dict:
            data = {
                key: emb.embed_dict(recurse_dict=recurse_dict, with_tags=with_tags) 
                for key, emb in self.features.items()
            }
            data = {
                **data,
                **{
                    f"{in_key}.{k}": v 
                    for in_key, in_emb in self.in_features.items() for k, v in in_emb.embed_dict(recurse_dict=recurse_dict, with_tags=with_tags).items()
                }
            }
        else:
            raise NotImplementedError
            data = {
                key: emb.embed() 
                for key, emb in itertools.chain(self.features.items(), self.in_features.items())
            }
            data = {
                **data,
                **{
                    f"{in_key}.{k}": v 
                    for in_key, in_emb in self.in_features.items() for k, v in in_emb.embed()
                }
            }
        
        if with_tags:
            assert recurse_dict, "Cannot add tags when not recursing dict"
            data = self.add_tags(data)
        
        return data

""" CONCRETE CLASSES """


# todo: doesn't use this yet
_num_move_values = len(MOVE_DICT)
def one_hot_move(move: Move) -> List[int]:
    ref_dict = MOVE_DICT
    if move.is_empty:
        value = -1
    else:
        assert move.id in ref_dict, f"Move not found: {move.id}"
        value = ref_dict[move.id]

    arr = [0] * _num_move_values
    if value >= 0:
        arr[value] = 1
    return arr

""" end sequence """
class MoveEmbed(AbstractEndEmbed):
    def __init__(self, move: Move):
        self.tags = {
            "move_name": move.id,
        }
        self.move = move
        # print("current_pp", move.entry)

    def embed_raw_dict(self) -> Dict:
        return {
            "move_id": -1,
            "acc": self.move.accuracy,
            "base_power": self.move.base_power,
            "current_pp": self.move.current_pp, # note: this number is not accurate
            "priority": self.move.priority,
            "is_unknown": int(self.move.is_empty),
        }

# todo: doesn't use this yet
_num_item_values = len(ITEM_DICT)
def one_hot_item(item: str) -> List[int]:
    ref_dict = ITEM_DICT
    if item == GEN8_DATA.UNKNOWN_ITEM:
        value = -1
    else:
        assert item in ref_dict, f"Item not found: {item}"
        value = ref_dict[item]

    arr = [0] * _num_item_values
    if value >= 0:
        arr[value] = 1
    return arr

""" end sequence """
class ItemEmbed(AbstractEndEmbed):
    def __init__(self, item: str):
        self.tags = {
            "item_name": item
        }
        self.item = item

    def embed_raw_dict(self) -> Dict:
        return {
            "item_id": one_hot_item(self.item),
            "is_unknown": int(self.item == GEN8_DATA.UNKNOWN_ITEM)
        }


# todo: doesn't use this yet
_num_ability_values = len(ABILITY_DICT)
def one_hot_ability(ability: str) -> List[int]:
    ref_dict = ABILITY_DICT
    if ability is None:
        value = -1
    else:
        assert ability in ref_dict, f"Ability not found: {ability}"
        value = ref_dict[ability]

    arr = [0] * _num_ability_values
    if value >= 0:
        arr[value] = 1
    return arr


""" end sequence """
class AbilityEmbed(AbstractEndEmbed):
    def __init__(self, ability: str):
        self.tags = {
            "ability_name": ability
        }
        self.ability = ability

    def embed_raw_dict(self) -> np.ndarray:
        return {
            "ability_id": one_hot_ability(self.ability),
            "is_unknown": int(self.ability is None)
        }
    

class EmptyStatus():
    def __init__(self):
        self.name = "NONE"
        self.value = -1

_num_status_values = max(Status, key=lambda x : x.value).value + 1
def one_hot_status(status: Status) -> List[int]:
    value = status.value
    arr = [0] * _num_status_values
    if value >= 0:
        arr[value] = 1
    return arr

class StatusEmbed(AbstractEndEmbed):
    def __init__(self, status: Status):
        self.is_none = status is None
        if status is None:
            status = EmptyStatus()

        self.tags = {
            "status_name": status.name,
            "status_id.idx": status.value,
        }
        self.status = status

    def embed_raw_dict(self) -> np.ndarray:
        return {
            "is_none": int(self.is_none),
            "status_id": one_hot_status(self.status),
        }


""" end sequence """
class PokemonDataEmbed(AbstractEndEmbed):
    def __init__(self, pok: Pokemon):
        self.tags = {
        }
        self.pok = pok

    def embed_raw_dict(self) -> np.ndarray:
        return {
            "pok_id": -1,
            "is_unknown": int(self.pok.is_empty),
        }

class MovesetEmbed(AbstractHLEmbed):
    def __init__(self, moveset: List[Move]):
        assert len(moveset) <= NUM_MOVES_PER_POKEMON
        moveset = moveset + [EmptyMove(move_id="empty")] * (NUM_MOVES_PER_POKEMON - len(moveset))

        self.features = {
            f"move{i}": MoveEmbed(move)
            for i, move in enumerate(moveset)
        }
        self.in_features = {}
        self.tags = {}
    
class PokemonEmbed(AbstractHLEmbed):
    def __init__(self, pok: Pokemon):
        self.features = {
            f"moveset": MovesetEmbed(list(pok.moves.values())),
            "status": StatusEmbed(pok.status),
            "ability": AbilityEmbed(pok.ability),
            "item": ItemEmbed(pok.item),
        }
        self.in_features = {
            "pok_data": PokemonDataEmbed(pok), 
        }
        self.tags = {
            "pok_name": pok.species,
        }
        self.pok = pok

class TeamEmbed(AbstractHLEmbed):
    def __init__(self, team: List[Pokemon]):
        if len(team) <= NUM_POKEMON_PER_TEAM:
            team = team + [EmptyPokemon()] * (NUM_POKEMON_PER_TEAM - len(team))

        self.features = {
            f"pok{i}": PokemonEmbed(pok)
            for i, pok in enumerate(team)
        }
        self.in_features = {}
        self.tags = {}

class BattleEmbed(AbstractHLEmbed):
    def __init__(self, battle: Battle):
        self.features = {
            "active_pokemon": PokemonEmbed(battle.active_pokemon),
            "opp_active_pokemon": PokemonEmbed(battle.opponent_active_pokemon),
            # "my_team": TeamEmbed(list(battle.team.values())),
            # "opp_team": TeamEmbed(list(battle.opponent_team.values())),
        }
        self.in_features = {}
        self.tags = {
            "turn": battle.turn,
            "my_player": battle.player_username,
            "opp_player": battle.opponent_username,
        }



class EmptyPokemon(Pokemon):
    def __init__(self, gen=8):

        self._ability = None
        self._item = GEN8_DATA.UNKNOWN_ITEM
        self._active = False
        self._gender = None
        self._level = None
        self._current_hp = None
        self._max_hp = 1 # not 0, to avoid div-by-zero issues
        self._moves = dict()
        self._species = EMPTY_SPECIES

        self._is_empty = True

