import asyncio

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers.legacy import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    Player,
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
    wrap_for_old_gym_api,
)
from poke_env.data import GenData

import sys 
sys.path.append("/Users/richardzhan/cs/15888/poke/python")

from rzlib.env import embed
from rzlib.env.old_rl_player import OldRLPlayer

_battle_format = "gen8ou"
_team1_fname = "../data/small_teams/team1.txt"
_team2_fname = "../data/small_teams/team2.txt"
_input_len = 380

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())

def create_random_player():
    return RandomPlayer(battle_format=_battle_format, team=load_team(_team1_fname))

def create_max_power_player():
    return MaxBasePowerPlayer(battle_format=_battle_format, team=load_team(_team1_fname))

def create_rl_player(*, opponent, **kwargs):
    return OldRLPlayer(
        input_len=_input_len,
        battle_format=_battle_format,
        opponent=opponent,
        start_challenging=True,
        team=load_team(_team2_fname),
        **kwargs
    )

def create_rl_env_random(sanity_check=True):
    print("create_rl_env_random()...")
    if sanity_check:
        # First test the environment to ensure the class is consistent
        # with the OpenAI API
        sanity_check_env = create_rl_player(opponent=create_random_player())
        check_env(sanity_check_env)
        sanity_check_env.close()

    # Create one environment for training and one for evaluation
    train_env = create_rl_player(opponent=create_random_player())
    train_env = wrap_for_old_gym_api(train_env)

    eval_env = create_rl_player(opponent=create_random_player())
    eval_env = wrap_for_old_gym_api(eval_env)

    return train_env, eval_env


def create_model(train_env, memory_limit=10000, anneal_steps=10000):
    print("create_model()...")
    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=memory_limit, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.01,
        value_test=0.0,
        nb_steps=anneal_steps,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.99,

        batch_size=1000,
        train_interval=1000,
        target_model_update=1000,
        # delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.001), metrics=["mae"])


    return dqn

def train_model(*, dqn: DQNAgent, train_env, num_steps=10000):
    print("train_model()...", num_steps)
    dqn.fit(train_env, nb_steps=num_steps)
    # train_env.close()

def eval_model(*, agent: DQNAgent, eval_env, opponent: Player, num_eval_episodes=100):
    print("eval_model()...")

    eval_env.reset_env(restart=True, opponent=opponent)
    print(f"Results against {opponent.username}:")
    agent.test(eval_env, nb_episodes=num_eval_episodes, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

import time
if __name__ == "__main__":
    start_time = time.time()

    train_env, eval_env = create_rl_env_random()

    # create the model
    dqn = create_model(
        train_env=train_env,
        anneal_steps=10000,
        memory_limit=100000,
    )

    # Training the model
    for epoch in range(10):
        print(f"epoch {epoch}...")

        # train_env, _ = create_rl_env_random()
        train_model(
            dqn=dqn,
            train_env=train_env,
            num_steps=10000,
        )

        # Evaluating the model
        ## recreate eval_env (sometimes breaks if the training takes too long)
        _, eval_env = create_rl_env_random()
        eval_model(
            agent=dqn,
            eval_env=eval_env,
            opponent=create_random_player(),
            num_eval_episodes=100,
        )

        print("time", time.time() - start_time)
