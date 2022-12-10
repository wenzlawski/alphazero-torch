import itertools
import os
import logging
import time
import torch
from torch.utils.data import DataLoader, RandomSampler

from alpha_zero.utils import spawn, stats, data_logger
from alpha_zero.learner.replay_data import ReplayData
from alpha_zero.learner.utils import watcher, Buffer, _init_model_from_config
from alpha_zero.torch import network


def save_checkpoint(path, step, model, optimizer=None, replaybuf=None):
    """Save model (and replay buffer) checkpoint
    :param name: File base name
    :param replaybuf: Save also replay buffer
    """
    state = {"model": model.state_dict(), "step": step}
    if optimizer:
        state["optimizer"] = optimizer.state_dict()
    path = os.path.join(path, f"checkpoint-{step}.policy.pth")
    torch.save(state, path)
    logging.info(f"saved policy checkpoint to {path}")

    # if replaybuf:
    #     rpath = f"checkpoint-{step}.replaybuf.pth"
    #     torch.save(replaybuf.state_dict(), rpath)
    #     logging.info(f"saved replay buffer checkpoint to {rpath}")
    return path


def _load_model_from_checkpoint(state_dict):
    """Load a model from a checkpoint"""
    model = model.Model()
    model.load_state_dict(state_dict)
    return model


@watcher
def learner(
    *, game, config, actors, evaluators, broadcast_fn, logger, model=None, s_step=None
):
    """A learner that consumes the replay buffer and trains the network."""
    logger.also_to_stdout = True
    replay_buffer = Buffer(config.replay_buffer_size)

    # * Number of distinct states being collected
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    logger.print("Initializing model")
    if model is not None and s_step is not None:
        model = _load_model_from_checkpoint(model)
    else:
        model = _init_model_from_config(config)
    logger.print(
        "Model type: %s(%s, %s)" % ("resnet", config.nn_width, config.nn_depth)
    )
    logger.print("Model size:", model.num_trainable_variables, "variables")
    save_path = save_checkpoint(
        config.path, 0 if s_step is None else s_step, model, model.optimizer
    )
    logger.print("Initial checkpoint:", save_path)
    broadcast_fn(save_path)

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
    evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
    total_trajectories = 0

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_trajectories():
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            replay_buffer.extend(
                network.TrainInput(s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states
            )

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []

        dataset = ReplayData(network.TrainInput.stack(replay_buffer.data))

        # * Using a random sampler to reuse n states in the replay buffer
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=config.replay_buffer_size
        )

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.train_batch_size,
        )

        for batch in dataloader:
            _, loss = model.update(batch, device=config.device)
            losses.append(loss)

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        s = step if step % config.checkpoint_freq == 0 else -1
        save_path = save_checkpoint(config.path, s, model, model.optimizer)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

    last_time = time.time() - 60
    for step in itertools.count(1 if s_step is None else s_step):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print("Step:", step)
        logger.print(
            (
                "Collected {:5} states from {:3} games, {:.1f} states/s. "
                "{:.1f} states/(s*actor), game length: {:.1f}"
            ).format(
                num_states,
                num_trajectories,
                num_states / seconds,
                num_states / (config.actors * seconds),
                num_states / num_trajectories,
            )
        )
        logger.print(
            "Buffer size: {}. States seen: {}".format(
                len(replay_buffer), replay_buffer.total_seen
            )
        )

        save_path, losses = learn(step)

        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write(
            {
                "step": step,
                "total_states": replay_buffer.total_seen,
                "states_per_s": num_states / seconds,
                "states_per_s_actor": num_states / (config.actors * seconds),
                "total_trajectories": total_trajectories,
                "trajectories_per_s": num_trajectories / seconds,
                "queue_size": 0,  # Only available in C++.
                "game_length": game_lengths.as_dict,
                "game_length_hist": game_lengths_hist.data,
                "outcomes": outcomes.data,
                "value_accuracy": [v.as_dict for v in value_accuracies],
                "value_prediction": [v.as_dict for v in value_predictions],
                "eval": {
                    "count": evals[0].total_seen,
                    "results": [sum(e.data) / len(e) if e else 0 for e in evals],
                },
                "batch_size": batch_size_stats.as_dict,
                "batch_size_hist": [0, 1],
                "loss": losses,
                "cache": {  # Null stats because it's hard to report between processes.
                    "size": 0,
                    "max_size": 0,
                    "usage": 0,
                    "requests": 0,
                    "requests_per_s": 0,
                    "hits": 0,
                    "misses": 0,
                    "misses_per_s": 0,
                    "hit_rate": 0,
                },
            }
        )
        logger.print()

        if config.max_steps > 0 and step >= config.max_steps:
            break

        broadcast_fn(save_path)
