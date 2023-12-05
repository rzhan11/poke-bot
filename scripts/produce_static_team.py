from pathlib import Path
import json
from typing import List
import re
import os

_gen = 8

data_folder = Path("../poke-env/src/poke_env/data/static")
output_folder = Path("../data/small_ref_dicts/")

team_folder = Path("../data/small_teams/")

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



def strip_word(word):
    import re
    word = word.replace("(F)", "").replace("(M)", "").strip()
    word = word.lower()
    word = re.sub(r'\W+', '', word)
    return word


print(os.listdir(team_folder))
team_species = set()
team_moves = set()
team_items = set()
team_abilities = set()
for fname in os.listdir(team_folder):
    with open(team_folder / fname, "r") as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            if "@" in line:
                words = line.split("@")
                sp = strip_word(words[0])
                print("Species", sp)
                team_species.add(sp)

                ## get items
                item = strip_word(words[1])
                print("Item", item)
                team_items.add(item)
            elif line.startswith("-"):
                move = strip_word(line[2:])
                team_moves.add(move)
                print("Move", move)
            elif line.startswith("Ability: "):
                ability = strip_word(line[9:])
                team_abilities.add(ability)
                print("Ability", ability)

for (team_v, all_v, v_name) in [(team_species, species, "species"), (team_moves, moves, "moves"), (team_items, items, "items"), (team_abilities, abilities, "abilities")]:
    for v in team_v:
        assert v in all_v, f"{v_name}: {v} is not found"

moves = team_moves
items = team_items
species = team_species
abilities = team_abilities



with open(output_folder / f"move.json", "w") as f:
    json.dump(produce_output_dict(moves), f, indent=2)
with open(output_folder / f"ability.json", "w") as f:
    json.dump(produce_output_dict(abilities), f, indent=2)
with open(output_folder / f"species.json", "w") as f:
    json.dump(produce_output_dict(species), f, indent=2)
with open(output_folder / f"item.json", "w") as f:
    json.dump(produce_output_dict(items), f, indent=2)