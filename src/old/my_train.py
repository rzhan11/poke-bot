import asyncio

import numpy as np
from gym.spaces import Box, Space
from gym.utils.env_checker import check_env
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

from ...python.rzlib.env import embed

_input_len = 11652

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: GenData = GenData.from_gen(8)

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=10.0, hp_value=1.0, victory_value=100.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        emb = embed.BattleEmbed(battle)

        emb_dict = emb.embed_dict()
        emb_arr = embed.convert_embed_dict_to_ndarray(emb_dict)

        # Final vector with 10 components
        return emb_arr

    def describe_embedding(self) -> Space:
        return Box(
            low=-100,
            high=1000,
            shape=(_input_len,),
            dtype=np.float32,
        )

def create_random_player():
    return RandomPlayer(battle_format="gen8randombattle")

def create_max_power_player():
    return MaxBasePowerPlayer(battle_format="gen8randombattle")

def create_rl_env_random(sanity_check=True):
    print("create_rl_env_random()...")
    if sanity_check:
        # First test the environment to ensure the class is consistent
        # with the OpenAI API
        sanity_check_env = SimpleRLPlayer(
            battle_format="gen8randombattle", 
            opponent=RandomPlayer(battle_format="gen8randombattle"),
            start_challenging=True, 
        )
        check_env(sanity_check_env)
        sanity_check_env.close()

    # Create one environment for training and one for evaluation
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", 
        opponent=create_random_player(), 
        start_challenging=True
    )
    train_env = wrap_for_old_gym_api(train_env)

    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", 
        opponent=create_random_player(), # dummy player
        start_challenging=True
    )
    eval_env = wrap_for_old_gym_api(eval_env)

    return train_env, eval_env


def create_model(train_env, nb_steps=10000):
    print("create_model()...")
    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=nb_steps, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=nb_steps,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=100,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])


    return dqn

def train_model(*, dqn: DQNAgent, train_env):
    print("train_model()...")
    dqn.fit(train_env, nb_steps=dqn.policy.nb_steps)
    train_env.close()

def eval_model(*, agent: DQNAgent, eval_env, opponent: Player, num_eval_episodes=100):
    print("eval_model()...")

    eval_env.reset_env(restart=True, opponent=opponent)
    print(f"Results against {opponent.username}:")
    agent.test(eval_env, nb_episodes=num_eval_episodes, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

async def main():
    train_env, eval_env = create_rl_env_random()

    # create the model
    dqn = create_model(
        train_env=train_env,
        nb_steps=50,
    )

    # Training the model
    train_model(
        dqn=dqn, 
        train_env=train_env,
    )

    # Evaluating the model
    eval_model(
        agent=dqn,
        eval_env=eval_env,
        opponent=create_random_player(),
        num_eval_episodes=100,
    )
    eval_model(
        agent=dqn,
        eval_env=eval_env,
        opponent=create_max_power_player(),
        num_eval_episodes=100,
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())