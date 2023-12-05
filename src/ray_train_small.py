# standard imports
import asyncio
import numpy as np
import json
import time
from pathlib import Path
import os

## gym imports
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env

## poke-env imports
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment import Move
from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)

## import RZLib
import sys 
sys.path.append("/Users/richardzhan/cs/15888/poke/python")

from rzlib.env.simple_rl_player import (
    SimpleRLPlayer,
    create_player_fn,
    create_random_bot_fn,
    create_max_bot_fn,
    create_heur_bot_fn,
)

## ray imports
import ray
_num_cpus = 4
_num_gpus = 0
ray.init(
    # local_mode=True,
    num_cpus=_num_cpus,
    num_gpus=_num_gpus,
    include_dashboard=True,
    dashboard_host="127.0.0.1",
    dashboard_port=6100,
    ignore_reinit_error=True,
)

## register the environment
from ray.tune.registry import register_env
select_env = "simple-rl-player-poke"
register_env(select_env, lambda config: SimpleRLPlayer(config))

## create PPO algo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.vector_env import VectorEnv, VectorEnvWrapper
from ray.rllib.evaluation import Episode, RolloutWorker


_battle_format = "gen8ou"
# _battle_format = "gen8randombattle"
_team1_fname = "../data/team1.txt"
_team2_fname = "../data/team2.txt"

def load_team(fname):
    # load_bot()
    with open(fname, "r") as f:
        return "".join(f.readlines())

def get_checkpoint_folder_version(folder_name):
    i = 0
    while True:
        path = Path(f"{folder_name}_{i}")
        if path.exists():
            i += 1
        else:
            break
    return path

def create_save_folder(base, cur_iter):
    return base / f"iter_{cur_iter:05d}/"

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: VectorEnvWrapper,
        policies,
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # print("on_episode_start", type(episode), episode.length)
        
        vec_env = base_env.vector_env
        # print("base_env", vec_env.num_envs, len(vec_env.envs))

        assert vec_env.num_envs == 1, f"Currently only supports single-env workers, tried to use {vec_env.num_envs}"
        
        cur_env = vec_env.envs[0]
        # print("n_won", type(cur_env), cur_env.n_won_battles, cur_env.n_finished_battles)
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     "after env reset!"
        # )

        episode.user_data["n_won_battles"] = cur_env.n_won_battles
        episode.user_data["n_finished_battles"] = cur_env.n_finished_battles

    def on_episode_end(
            self, 
            *, 
            worker: RolloutWorker, 
            base_env: VectorEnvWrapper, 
            policies, 
            episode: Episode, 
            env_index: int, 
            **kwargs
        ):
        # print("on_episode_end")
        cur_env = base_env.vector_env.envs[0]

        assert episode.user_data["n_finished_battles"] + 1 == cur_env.n_finished_battles
        won_battle = cur_env.n_won_battles - episode.user_data["n_won_battles"]
        assert 0 <= won_battle <= 1, f"won_battle is not 0/1, received {won_battle}"
        
        # print("won_battle", won_battle, "n_fin", episode.user_data["n_won_battles"], episode.user_data["n_finished_battles"])
        episode.custom_metrics["won_battle"] = won_battle
        # episode.custom_metrics["episode_id"] = episode.episode_id

    def on_sample_end(self, *, worker, samples, **kwargs):
        # print(f"on_sample_end, {len(samples)}")
        pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["custom_metrics"]["winrate"] = np.mean(result["custom_metrics"]["won_battle"])
        result["custom_metrics"]["n_won_battles"] = np.sum(result["custom_metrics"]["won_battle"]).astype(float)
        result["custom_metrics"]["n_finished_battles"] = len(result["custom_metrics"]["won_battle"])

        del result["custom_metrics"]["won_battle"]


# def custom_evaluation_function(algo, eval_workers):
#     from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
#     # eval_workers.foreach_worker(
#     #     func = lambda w: w.foreach_env(
#     #         lambda env: 
#     #     )
#     # )

#     for i in range(5):

#         def foreach_fn(w):
#             w.sample()
#             return f"i: {i}, {type(w)}"
#         print("Custom evaluation round", i)
#         res = eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)
#         print(res)

#     episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
#     # You can compute metrics from the episodes manually, or use the
#     # convenient `summarize_episodes()` utility:
#     metrics = summarize_episodes(episodes)

#     # You can also put custom values in the metrics dict.
#     metrics["foo"] = 1
#     return metrics


# num_eval_workers: https://discuss.ray.io/t/num-gpu-rollout-workers-learner-workers-evaluation-workers-purpose-resource-allocation/10159

_num_eval_workers = 0 ## This is probably not useful... (i think? see above link)
_num_rollout_workers = int(1 * (_num_cpus - _num_eval_workers))
_num_envs_per_worker = 1
_num_workers = _num_rollout_workers + _num_eval_workers

ppo_config = (
    PPOConfig()
    .framework("torch")
    .training(
        model={
            "fcnet_hiddens": [128, 128],
            # "use_lstm": True,
            # "lstm_cell_size": 64,
        },
        train_batch_size=1024,
        sgd_minibatch_size=1024,
        # num_sgd_iters=100,
        gamma=0.999,
        lr=0.001,
        use_critic=True,
        use_gae=True,
        shuffle_sequences=True,
        # kl_coeff=0.3,
        # target_network_update_freq=500,
        # gamma=0.99,
        # lr=0.001,
        # grad_clip=0.1,
        # optimizer=None,
        # train_batch_size=None,
    )
    .rollouts(
        num_rollout_workers=_num_rollout_workers,
        num_envs_per_worker=_num_envs_per_worker,
    )
    .evaluation(
        # evaluation_num_workers=_num_eval_workers,
        # evaluation_interval=1,
        # evaluation_parallel_to_training=True,
        # evaluation_duration="auto",
        # custom_evaluation_function=custom_evaluation_function,
    )
    .callbacks(
        MyCallbacks
    )
    .reporting(
        keep_per_episode_custom_metrics=True,
        metrics_num_episodes_for_smoothing=1000,
    )
    .resources(
        num_cpus_per_worker=_num_cpus / _num_workers, 
        num_gpus_per_worker=_num_gpus / _num_workers,
    )
    .environment(env=select_env, env_config={
        "base_config": {
            "battle_format": _battle_format,
            "start_challenging": True,
            "start_timer_on_battle_start": True,
            # "team": None, # my team
            "team": load_team(_team2_fname), # my team
        },
        "ray_config": {
            "opponent_fn": create_random_bot_fn(
                battle_format=_battle_format, 
                team=load_team(_team1_fname),
                # team=None,
            ),
            "base_name": "PPO",
        },
        "custom_config": {
            # "input_len": 514,
            # "input_len": 3598,
            "input_len": 29880,
        },
    })
    .fault_tolerance(
        recreate_failed_workers=True,
        max_num_worker_restarts=100,
        delay_between_worker_restarts_s=10,
        num_consecutive_worker_failures_tolerance=3,
    )
)

_num_iters = 10000
_checkpoint_freq = 10
_checkpoint_base_folder_name = "../results/ppo_toy"
_checkpoint_folder = get_checkpoint_folder_version(_checkpoint_base_folder_name)

# _use_checkpoint = "../results/ppo_1/"
_use_checkpoint = None
_checkpoint_iter = 350

if _use_checkpoint is not None: ## Load from a checkpoint
    from ray.rllib.algorithms.algorithm import Algorithm

    assert type(_use_checkpoint) is str, f"_use_checkpoint should be a str, received {type(_use_checkpoint)}"
    _checkpoint_folder = Path(_use_checkpoint)
    algo = Algorithm.from_checkpoint(
        create_save_folder(
            base=_checkpoint_folder, 
            cur_iter=_checkpoint_iter,
        )
    )

    print("Using checkpoint", _use_checkpoint)
    start_iter = _checkpoint_iter + 1
else:
    algo = ppo_config.build()
    start_iter = 0

print(json.dumps({
    "_num_cpus": _num_cpus,
    "_num_gpus": _num_gpus,
    "_num_rollout_workers": _num_rollout_workers,
}, indent=2))


orig_start_time = time.time()
_checkpoint_folder.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint folder {_checkpoint_folder}")
for cur_iter in range(start_iter, _num_iters):
    print("\n\nIter", cur_iter)

    train_start_time = time.time()
    train_res = algo.train()
    print("train", json.dumps({
        "training_iteration": train_res["training_iteration"],
        "episode_reward_mean": train_res["sampler_results"]["episode_reward_mean"],

        "num_env_steps_sampled_this_iter": train_res["num_env_steps_sampled_this_iter"],
        "episodes_this_iter": train_res["sampler_results"]["episodes_this_iter"],
        
        "timesteps_total": train_res["timesteps_total"],
        "episodes_total": train_res["episodes_total"],
        
        "counters": train_res["counters"],
        
        "timers": train_res["timers"],
        "time_this_iter_s": train_res["time_this_iter_s"],
        "time_total_s": train_res["time_total_s"],
        "custom_metrics": train_res["custom_metrics"],
    }, indent=2, default=lambda x: f"<convert {type(x)}>: {str(x)}"))
    print(f"Train time: {(time.time() - train_start_time):.0f} s")

    # checkpoint the algo (every X intervals + on the last iteration)
    if cur_iter % _checkpoint_freq == 0 or cur_iter == _num_iters - 1:
        save_folder = create_save_folder(_checkpoint_folder, cur_iter)
        # print(f"Checkpointing to {save_path}")
        save_res = algo.save(checkpoint_dir=save_folder)
        print("Saved", save_res.checkpoint.path)
        
        # save train_metrics also
        with open(save_folder / "train_res.json", "w") as f:
            json.dump(train_res, f, indent=2, default=lambda x: f"<convert {type(x)}>: {str(x)}")

    # eval_start_time = time.time()
    # eval_res = algo.evaluate()
    # print("eval_reward", eval_res["evaluation"]["episode_reward_mean"])
    # print(f"Eval time: {(time.time() - eval_start_time):.0f} s")

    # print(f"Total time: {(time.time() - orig_start_time):.0f} s")
