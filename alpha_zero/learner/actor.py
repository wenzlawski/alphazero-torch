import itertools
from alpha_zero.learner.utils import (
    watcher,
    _init_model_from_config,
    _init_bot,
    update_checkpoint,
    _play_game,
)
import alpha_zero.torch.evaluator as evaluator_lib


@watcher
def actor(*, config, game, logger, queue):
    """An actor process runner that generates games and returns trajectories."""
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")
    az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    bots = [
        _init_bot(config, game, az_evaluator, False),
        _init_bot(config, game, az_evaluator, False),
    ]
    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, az_evaluator):
            return
        queue.put(
            _play_game(
                logger,
                game_num,
                game,
                bots,
                config.temperature,
                config.temperature_drop,
            )
        )
