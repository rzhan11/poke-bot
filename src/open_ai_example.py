import numpy as np
from gym.spaces import Space, Box
from gym.core import ObsType
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment import AbstractBattle, Move
from poke_env.data import GenData

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data = GenData.from_gen(8)

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        """ 
        Indexes 0-3: base powers of our 4 moves
        Indexes 4-7: damage multipliers of our 4 moves
        Indexes 8: how many fainted pokemon on our team
        Indexes 9: how many fainted pokemon on their team
        """
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)

        move: Move
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=self._data.type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        """ Provide a range of values for the observation """
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


def do_dqn(train_env, eval_env):
    import tensorflow as tf
    ## Hacky workaround for "tf.keras.__version__" issue
    from keras import __version__
    tf.keras.__version__ = __version__

    """ Define the model """
    # imports for the model
    from tensorflow import keras
    from keras.layers import Dense, Flatten
    from keras.models import Sequential
    # imports for the agent
    from rl.agents.dqn import DQNAgent
    from rl.memory import SequentialMemory
    from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
    from keras.optimizers import Adam

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape # (1,) is the batch size that the model expects in input.

    print("create_dqn: input_shape", input_shape)

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    """ Define the Agent (memory, policy, optimizer) """

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    """ Train the model """
    dqn.fit(train_env, nb_steps=10000)
    train_env.close()

    """ Evaluate the model (against random and max damage players) """
    # random player
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    
    # max damage player
    from max_damage_player import MaxDamagePlayer
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max damage player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

def main():
    """ Validate the gym environment """
    from gym.utils.env_checker import check_env
    from poke_env.player import RandomPlayer

    opponent = RandomPlayer(battle_format="gen8randombattle")
    sanity_check_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    check_env(sanity_check_env)
    sanity_check_env.close()

    """ Setup the environment (including the random opponent) """
    from poke_env.player import RandomPlayer, wrap_for_old_gym_api

    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    train_env = wrap_for_old_gym_api(train_env)
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    eval_env = wrap_for_old_gym_api(eval_env)

    """ Setup the environment (including the random opponent) """
    do_dqn(train_env, eval_env)

if __name__ == "__main__":
    main()