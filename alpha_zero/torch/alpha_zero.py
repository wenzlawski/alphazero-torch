"""
This file will contain the training logic.

We want to implement:
- multi-processing for actors and evaluators
-

"""

import datetime
import json
import os
import sys
import tempfile

from alpha_zero.utils import spawn
from alpha_zero.learner.utils import Config, load_checkpoint, update_config
from alpha_zero.learner.actor import actor
from alpha_zero.learner.evaluator import evaluator
from alpha_zero.learner.learner import learner
import pyspiel


# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


def alpha_zero(config: Config, load=False):
    """Start all the worker processes for a full alphazero setup."""

    if load:
        cm = config
        checkpoint = load_checkpoint(config.path)
        config = checkpoint["config"]
        model = checkpoint["model"]
        step = checkpoint["step"]
        config = update_config(config, cm)

    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
    )

    print("Starting game", config.game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can only handle 2-player games.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("Game must be deterministic.")

    path = config.path
    if not path:
        path = tempfile.mkdtemp(
            prefix="az-{}-{}-".format(
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game
            )
        )
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))
    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % ("resnet", config.nn_width, config.nn_depth))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    actors = [
        spawn.Process(actor, kwargs={"game": game, "config": config, "num": i})
        for i in range(config.actors)
    ]
    evaluators = [
        spawn.Process(evaluator, kwargs={"game": game, "config": config, "num": i})
        for i in range(config.evaluators)
    ]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
        learner(
            game=game,
            config=config,
            actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators,
            broadcast_fn=broadcast,
            model=model if load else None,
            s_step=step if load else None,
        )
    except (KeyboardInterrupt, EOFError):
        # Save buffer, cache
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        broadcast("")
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(JOIN_WAIT_DELAY)
        for proc in evaluators:
            proc.join()
