from typing import List, Dict, Any, Tuple, Optional

from poke_env.data import GenData
from poke_env.environment import Battle
from poke_env.environment import Pokemon, PokemonType
from poke_env.environment import Status, Effect
from poke_env.environment import Move, EmptyMove, MoveCategory

import itertools
import numpy as np
from pathlib import Path
import json

## HYPERPARAMS
_cur_gen = 8
_static_data_folder = Path(__file__).parent / "../../../data/"
_move_fpath = _static_data_folder / "move.json"
_item_fpath = _static_data_folder / "item.json"
_ability_fpath = _static_data_folder / "ability.json"
_species_fpath = _static_data_folder / "species.json"


_cur_battle, _cur_turn = -1, -1
NUM_MOVES_PER_POKEMON = 4
NUM_POKEMON_PER_TEAM = 2

# constants for empty
EMPTY_SPECIES = "empty_species"
BASE_STAT_NAMES = ["atk", "def", "hp", "spa", "spd", "spe"]
BOOST_NAMES = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]

GEN8_DATA = GenData.from_gen(_cur_gen)
## init static DICTS (for one hot vector)
with open(_move_fpath, "r") as f:
    MOVE_DICT = json.load(f)
with open(_item_fpath, "r") as f:
    ITEM_DICT = json.load(f)
with open(_ability_fpath, "r") as f:
    ABILITY_DICT = json.load(f)
with open(_species_fpath, "r") as f:
    SPECIES_DICT = json.load(f)


def embed_dumps(z):
    import json
    from json import JSONEncoder
    import re

    # tag lists
    class MarkedList:
        _list = None
        def __init__(self, l):
            self._list = l

    def mark_lists(d):
        if type(d) is dict:
            d = {k: mark_lists(v) for k, v in d.items()}
        if type(d) is list or type(d) is tuple:
            d = MarkedList(d)
        return d
    
    z = mark_lists(z)

    class CustomJSONEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, MarkedList):
                return "##<{}>##".format(o._list)
        
    b = json.dumps(z, indent=2, separators=(',', ':'), cls=CustomJSONEncoder)
    b = b.replace('"##<', " ").replace('>##"', "")
    return b



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
    return np.concatenate(arr, dtype=np.float32)



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


_num_category_values = max(MoveCategory, key=lambda x : x.value).value
def one_hot_category(cat: MoveCategory) -> List[int]:
    arr = [0] * _num_category_values
    if cat is not None:
        value = cat.value - 1
        assert value >= 0
        arr[value] = 1
    return arr

""" end sequence """
class MoveEmbed(AbstractEndEmbed):
    def __init__(self, move: Move):
        self.tags = {
            "move_name": move.id,
            ".boosts": move.boosts,
            ".category": str(move.category),
            # ".secondary": move.secondary,
        }
        self.move = move
        # print("current_pp", move.entry)

    def embed_raw_dict(self) -> Dict:
        # encode boosts
        boosts = {name: 0 for name in BOOST_NAMES}
        if self.move.boosts is not None:
            for k, v in self.move.boosts.items():
                boosts[k] = v

        return {
            "is_unknown": int(self.move.is_empty),

            # "move_id": one_hot_move(self.move),
            "type": one_hot_type((self.move.type,)),
            "category": one_hot_category(self.move.category),
            "defensive_category": one_hot_category(self.move.defensive_category),
            "current_pp": self.move.current_pp, # note: this number is not accurate

            "accuracy": self.move.accuracy,
            "base_power": self.move.base_power,
            "boosts": list(boosts.values()),
            "priority": self.move.priority,
            "damage": 0.0 if self.move.damage == "level" else self.move.damage,
            "damage_use_level": int(self.move.damage == "level"),
            
            "drain": self.move.drain,
            "crit_ratio": self.move.crit_ratio,
            "force_switch": int(self.move.force_switch),
            "status": one_hot_status(self.move.status),
            "is_volatile_status": int(self.move.volatile_status is not None),
            "heal": self.move.heal,

            # "cure_status": self.move.
        }
    
def is_no_item(item: str):
    return item is None or item == ""

# todo: doesn't use this yet
_num_item_values = len(ITEM_DICT)
def one_hot_item(item: str) -> List[int]:
    ref_dict = ITEM_DICT
    if item == GEN8_DATA.UNKNOWN_ITEM or is_no_item(item):
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
            # "item_id": one_hot_item(self.item),
            "is_unknown": int(self.item == GEN8_DATA.UNKNOWN_ITEM),
            "is_none": int(is_no_item(self.item)),
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
            # "ability_id": one_hot_ability(self.ability),
            "is_unknown": int(self.ability is None)
        }
    

_num_status_values = max(Status, key=lambda x : x.value).value
def one_hot_status(status: Status) -> List[int]:
    arr = [0] * _num_status_values
    if status is not None:
        value = status.value - 1
        assert value >= 0
        arr[value] = 1
    return arr

class StatusEmbed(AbstractEndEmbed):
    def __init__(self, status: Status, status_counter: int):
        self.is_none = status is None

        self.tags = {
            "status_name": status.name if status is not None else "None",
            "status_id.idx": status.value if status is not None else "None",
        }
        self.status = status
        self.status_counter = status_counter

    def embed_raw_dict(self) -> np.ndarray:
        return {
            "is_none": int(self.is_none),
            "status_id": one_hot_status(self.status),
            "status_counter": self.status_counter,
        }



# todo: doesn't use this yet
_num_species_values = len(SPECIES_DICT)
def one_hot_species(pok: Pokemon) -> List[int]:
    ref_dict = SPECIES_DICT
    if pok.is_empty:
        value = -1
    else:
        assert pok.species in ref_dict, f"Species not found: {pok.species}"
        value = ref_dict[pok.species]

    arr = [0] * _num_species_values
    if value >= 0:
        arr[value] = 1
    return arr

def encode_effect(effects):
    if len(effects) == 0:
        return [0, 0]
    else:
        # return first effect + counter
        for k, v in effects.items():
            return [1, v]

_num_type_values = max(PokemonType, key=lambda x : x.value).value
def one_hot_type(types: Tuple[PokemonType]) -> List[int]:
    arr = [0] * _num_type_values
    for pok_type in types:
        if pok_type is not None:
            value = pok_type.value - 1
            assert value >= 0
            arr[value] = 1
    return arr


""" end sequence """
class PokemonDataEmbed(AbstractEndEmbed):
    def __init__(self, pok: Pokemon):
        self.tags = {
            # "base_stats": pok.base_stats,
            # "boosts": pok.boosts,
            "all_effects": {str(k.name): v for k, v in pok.effects.items()},
            "all_types": (str(pok.type_1), str(pok.type_2)),
            # "all_level": (str(type(pok.level))),
        }
        self.pok = pok

    def embed_raw_dict(self) -> np.ndarray:
        return {
            # meta info
            "is_unknown": isinstance(self.pok, UnknownPokemon),
            "revealed": int(self.pok.revealed),
            "active": int(self.pok.active),

            # poke-specific info
            # "species_id": one_hot_species(self.pok),
            "level": int(self.pok.level),
            "types": one_hot_type((self.pok.type_1, self.pok.type_2)),
            "base_stats": list(self.pok.base_stats.values()),
            "fainted": int(self.pok.fainted),
            "boosts": list(self.pok.boosts.values()),
            "cur_hp_fraction": self.pok.current_hp_fraction,
            # "max_hp": self.pok.max_hp, ### this might be bad to use, since our team has real values, while opp team is scaled 0 to 100
            "effects": encode_effect(self.pok.effects),
            "protect_counter": int(self.pok.protect_counter),
            "is_dyna": int(self.pok.is_dynamaxed),
        }

class MovesetEmbed(AbstractHLEmbed):
    def __init__(self, moveset: List[Move]):
        # tag all moves
        assert len(moveset) <= NUM_MOVES_PER_POKEMON
        moveset = moveset + [UnknownMove()] * (NUM_MOVES_PER_POKEMON - len(moveset))

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
            "status": StatusEmbed(pok.status, pok.status_counter),
            # "ability": AbilityEmbed(pok.ability),
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
            team = team + [UnknownPokemon()] * (NUM_POKEMON_PER_TEAM - len(team))

        self.features = {
            f"pok{i}": PokemonEmbed(pok)
            for i, pok in enumerate(team)
        }
        self.in_features = {}
        self.tags = {}

class BattleEmbed(AbstractHLEmbed):
    def __init__(self, battle: Battle):
        global _cur_turn, _cur_battle
        _cur_battle = battle.battle_tag
        _cur_turn = battle.turn

        self.features = {
            "active_pokemon": PokemonEmbed(battle.active_pokemon),
            "opp_active_pokemon": PokemonEmbed(battle.opponent_active_pokemon),
            # "my_team": TeamEmbed(list(battle.team.values())),
            # "opp_team": TeamEmbed(list(battle.opponent_team.values())),
        }
        self.in_features = {}
        self.tags = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "my_player": battle.player_username,
            "opp_player": battle.opponent_username,
        }


            # # meta info
            # "is_unknown": int(self.pok.is_empty),
            # "revealed": int(self.pok.revealed),
            # "active": int(self.pok.active),

            # # poke-specific info
            # # "species_id": one_hot_species(self.pok),
            # "level": self.pok.level,
            # "types": one_hot_type((self.pok.type_1, self.pok.type_2)),
            # "base_stats": list(self.pok.base_stats.values()),
            # "fainted": int(self.pok.fainted),
            # "boosts": list(self.pok.boosts.values()),
            # "cur_hp_fraction": self.pok.current_hp_fraction,
            # "effects": encode_effect(self.pok.effects),
            # "protect_counter": self.pok.protect_counter,
            # "is_dyna": int(self.pok.is_dynamaxed),

class UnknownPokemon(Pokemon):
    def __init__(self, gen=8):
        # in PokemonEmbed
        self._ability = None
        self._item = GEN8_DATA.UNKNOWN_ITEM
        self._status = None
        self._moves = dict()

        ## in PokemonDataEmbed

        # meta info
        self._active = False
        self._revealed = False
        self._gender = None

        # poke-specific info
        self._species = EMPTY_SPECIES
        self._level = 0
        self._type_1 = None
        self._type_2 = None
        self._base_stats = {stat_name: 0.0 for stat_name in BASE_STAT_NAMES}
        self._fainted = False
        self._boosts = {stat_name: 0.0 for stat_name in BOOST_NAMES}
        self._current_hp = None
        self._max_hp = 100 # not 0, to avoid div-by-zero issues
        self._effects = {}
        self._protect_counter = 0
        self._is_dynamaxed = False

        self._terastallized = False
        self._terastallized_type = None
        self._must_recharge = False



####

    #     self.tags = {
    #         "move_name": move.id,
    #         ".boosts": move.boosts,
    #         ".category": move.category,
    #     }
    #     self.move = move

    #     if not move.is_empty and move.boosts is not None:
    #         print("no boosts")
    #     # print("current_pp", move.entry)

    # def embed_raw_dict(self) -> Dict:
    #     # encode boosts
    #     boosts = self.move.boosts
    #     if self.move.is_empty or boosts is None:
    #         boosts = {name: 0 for name in BOOST_NAMES}

    #     return {
    #         # "move_id": one_hot_move(self.move),
    #         "accuracy": self.move.accuracy,
    #         "base_power": self.move.base_power,
    #         "boosts": list(boosts.values()),
    #         "category": one_hot_category(self.move.category),
    #         "defensive_category": one_hot_category(self.move.defensive_category),
    #         "current_pp": self.move.current_pp, # note: this number is not accurate
    #         "priority": self.move.priority,
    #         "is_unknown": int(self.move.is_empty),
    #     }

class UnknownMove(Move):
    def __init__(self, gen=8):
        self._gen = gen

        self._id = "unknown_move"

        self._move_id = "unknown_move"
        self._current_pp = 0
        
        self._is_empty = True

    @property
    def accuracy(self) -> float:
        return 1.0

    @property
    def base_power(self) -> int:
        return 0

    @property
    def category(self) -> MoveCategory:
        return None

    @property
    def crit_ratio(self) -> int:
        return 0

    @property
    def damage(self) -> int:
        return 0

    @property
    def defensive_category(self) -> MoveCategory:
        return None

    @property
    def drain(self) -> float:
        return 0.0

    @property
    def force_switch(self) -> bool:
        return False

    @property
    def heal(self) -> float:
        return 0.0

    @property
    def priority(self) -> int:
        return 0

    @property
    def secondary(self) -> List[Dict[str, Any]]:
        return []

    @property
    def status(self) -> Status:
        return None

    @property
    def type(self) -> PokemonType:
        return None

    @property
    def volatile_status(self) -> Optional[str]:
        return None

    @property
    def boosts(self) -> Optional[Dict[str, int]]:
        return {name: 0 for name in BOOST_NAMES}