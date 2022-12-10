import collections
from dataclasses import dataclass, field
import functools
import os
import random
import traceback
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

import alpha_zero.torch.model as model
from alpha_zero.utils import data_logger, file_logger, spawn, stats

#! Need replacing
from open_spiel.python.algorithms import mcts


@dataclass
class TrajectoryState(object):
    """A particular point along a trajectory."""

    observation: List[int]  # * [9, 11, 11] 1d array
    current_player: int  # * Duh
    legals_mask: List[int]  # * Legal moves mask
    action: int  # * Action taken
    policy: List[int]  # * Policy map based on MCTS
    value: List[int]  # * Value of the state


@dataclass
class Trajectory(object):
    """A sequence of observations, actions and policies, and the outcomes."""

    states: list = field(default_factory=list)
    returns: int = None

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


class Buffer(object):
    """A fixed size buffer that keeps the newest values."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.total_seen = 0  # The number of items that have passed through.

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val):
        return self.extend([val])

    def extend(self, batch):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[: -self.max_size] = []

    def sample(self, count):
        return random.sample(self.data, count)


class Config(
    collections.namedtuple(
        "Config",
        [
            "game",
            "device",
            "path",
            "learning_rate",
            "weight_decay",
            "l2_regularisation",
            "train_batch_size",
            "momentum",
            "replay_buffer_size",
            "replay_buffer_reuse",
            "max_steps",
            "checkpoint_freq",
            "actors",
            "evaluators",
            "evaluation_window",
            "eval_levels",
            "uct_c",
            "max_simulations",
            "policy_alpha",
            "policy_epsilon",
            "temperature",
            "temperature_drop",
            "nn_width",
            "nn_depth",
            "observation_shape",
            "output_size",
            "quiet",
        ],
    )
):
    """A config for the model/experiment."""

    pass


def update_config(old: Config, new: Config):
    """Update an old config. Only certain attributes can be updated"""
    return old._replace(
        device=new.device,
        path=os.path.dirname(new.path),
        learning_rate=new.learning_rate,
        weight_decay=new.weight_decay,
        l2_regularisation=new.l2_regularisation,
        momentum=new.momentum,
        train_batch_size=new.train_batch_size,
        replay_buffer_size=new.replay_buffer_size,
        replay_buffer_reuse=new.replay_buffer_reuse,
        max_steps=new.max_steps,
        checkpoint_freq=new.checkpoint_freq,
        actors=new.actors,
        evaluators=new.evaluators,
        uct_c=new.uct_c,
        max_simulations=new.max_simulations,
        policy_alpha=new.policy_alpha,
        policy_epsilon=new.policy_epsilon,
        temperature=new.temperature,
        temperature_drop=new.temperature_drop,
        evaluation_window=new.evaluation_window,
        eval_levels=new.eval_levels,
        quiet=new.quiet,
    )


def _init_model_from_config(config):
    az = model.Model()
    az.build_model(config._asdict())
    return az


def watcher(fn):
    """A decorator to fn/processes that gives a logger and logs exceptions."""

    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = fn.__name__
        if num is not None:
            name += "-" + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return fn(config=config, logger=logger, **kwargs)
            except Exception as e:
                logger.print(
                    "\n".join(
                        [
                            "",
                            " Exception caught ".center(60, "="),
                            traceback.format_exc(),
                            "=" * 60,
                        ]
                    )
                )
                print("Exception caught in {}: {}".format(name, e))
                raise
            finally:
                logger.print("{} exiting".format(name))
                print("{} exiting".format(name))

    return _watcher


def _init_bot(config, game, evaluator_, evaluation):
    """Initializes a bot."""
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    return mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True,
    )


def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions = []
    state = game.new_initial_state()
    random_state = np.random.RandomState()
    logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
    logger.opt_print("Initial state:\n{}".format(state))
    while not state.is_terminal():
        if state.is_chance_node():
            # For chance nodes, rollout according to chance node's probability
            # distribution
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random_state.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            root = bots[state.current_player()].mcts_search(state)
            policy = np.zeros(game.num_distinct_actions())
            for c in root.children:
                policy[c.action] = c.explore_count
            policy = policy ** (1 / temperature)
            policy /= policy.sum()
            if len(actions) >= temperature_drop:
                action = root.best_child().action
            else:
                action = np.random.choice(len(policy), p=policy)
            trajectory.states.append(
                TrajectoryState(
                    state.observation_tensor(),
                    state.current_player(),
                    state.legal_actions_mask(),
                    action,
                    policy,
                    root.total_reward / root.explore_count,
                )
            )
            action_str = state.action_to_string(state.current_player(), action)
            actions.append(action_str)
            logger.opt_print(
                "Player {} sampled action: {}".format(
                    state.current_player(), action_str
                )
            )
            state.apply_action(action)
    logger.opt_print("Next state:\n{}".format(state))

    trajectory.returns = state.returns()
    logger.print(
        "Game {}: Returns: {}; Actions: {}".format(
            game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)
        )
    )
    return trajectory


def load_checkpoint(path: str):
    checkpoint = torch.load(path)
    return checkpoint
    # config = Config(checkpoint["config"])
    # model = model.Model.load_state_dict(checkpoint["model"])
    # return config, model, checkpoint["step"]


def update_checkpoint(logger, queue, model, az_evaluator):
    """Read the queue for a checkpoint to load, or an exit signal."""
    path = None
    while True:  # Get the last message, ignore intermediate ones.
        try:
            path = queue.get_nowait()
        except spawn.Empty:
            break
    if path:
        logger.print("Inference cache:", az_evaluator.cache_info())
        logger.print("Loading checkpoint", path)
        state = load_checkpoint(path)["model"]
        model.load_state_dict(state)
        az_evaluator.clear_cache()
    elif path is not None:  # Empty string means stop this process.
        return False
    return True
