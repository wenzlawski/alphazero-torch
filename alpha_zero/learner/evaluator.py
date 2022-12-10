import itertools
import numpy as np

from alpha_zero.learner.utils import (
    Buffer,
    watcher,
    _init_model_from_config,
    _init_bot,
    update_checkpoint,
    _play_game,
)
import alpha_zero.torch.evaluator as evaluator_lib

#! Need replacing
import open_spiel.python.algorithms.mcts as mcts


@watcher
def evaluator(*, game, config, logger, queue):
    """A process that plays the latest checkpoint vs standard MCTS."""
    results = Buffer(config.evaluation_window)
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    random_evaluator = mcts.RandomRolloutEvaluator()

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return

        az_player = game_num % 2
        difficulty = (game_num // 2) % config.eval_levels
        max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
        bots = [
            _init_bot(config, game, az_evaluator, True),
            mcts.MCTSBot(
                game,
                config.uct_c,
                max_simulations,
                random_evaluator,
                solve=True,
                verbose=False,
                dont_return_chance_node=True,
            ),
        ]
        if az_player == 1:
            bots = list(reversed(bots))

        trajectory = _play_game(
            logger, game_num, game, bots, temperature=1, temperature_drop=0
        )
        results.append(trajectory.returns[az_player])
        queue.put((difficulty, trajectory.returns[az_player]))

        logger.print(
            "AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
                trajectory.returns[az_player],
                trajectory.returns[1 - az_player],
                len(results),
                np.mean(results.data),
            )
        )
