from pathlib import Path
import json
from typing import List
import re

_gen = 8

data_folder = Path("/Users/richardzhan/opt/miniconda3/envs/poke/lib/python3.10/site-packages/poke_env/data/static")
output_folder = Path("../data/")

def produce_output_dict(l: List):
    return {v: k for k, v in enumerate(l)}


# produce moves
with open(data_folder / f"moves/gen{_gen}moves.json", "r") as f:
    moves_obj = json.load(f)
    moves = list(moves_obj.keys())

# produce species
with open("../pokemon-showdown/data/learnsets.ts", "r") as f:
    lines = f.readlines()
    all_data = "\n".join(lines)
    species1 = re.findall(r"^\t(\w+): {$", all_data, re.MULTILINE)
with open("../pokemon-showdown/data/pokedex.ts", "r") as f:
    lines = f.readlines()
    all_data = "\n".join(lines)
    species2 = re.findall(r"^\t(\w+): {$", all_data, re.MULTILINE)
species = list(set(species1 + species2))

# produce abilities
with open("../pokemon-showdown/data/abilities.ts", "r") as f:
    lines = f.readlines()
    all_data = "\n".join(lines)
    abilities = re.findall(r"^\t(\w+): {$", all_data, re.MULTILINE)


# produce items
with open("../pokemon-showdown/data/items.ts", "r") as f:
    lines = f.readlines()
    all_data = "\n".join(lines)
    items = re.findall(r"^\t(\w+): {$", all_data, re.MULTILINE)

        


# produce species

with open(output_folder / f"move.json", "w") as f:
    json.dump(produce_output_dict(moves), f, indent=2)
with open(output_folder / f"ability.json", "w") as f:
    json.dump(produce_output_dict(abilities), f, indent=2)
with open(output_folder / f"species.json", "w") as f:
    json.dump(produce_output_dict(species), f, indent=2)
with open(output_folder / f"item.json", "w") as f:
    json.dump(produce_output_dict(items), f, indent=2)