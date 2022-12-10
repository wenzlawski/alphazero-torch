# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Starting point for playing with the AlphaZero algorithm."""

from absl import app

import alpha_zero.torch.alpha_zero as alpha_zero
from alpha_zero.utils import spawn

game = "hex"
load = False
path = "data/model2"
device = "cpu"
uct_c = 2
max_simulations = 20
train_batch_size = 2**4
replay_buffer_size = 2**6
replay_buffer_reuse = 4
learning_rate = 0.01
weight_decay = 0.1
l2_regularisation = 0.01
momentum = 0.01
policy_epsilon = 0.3
policy_alpha = 1
temperature = 1
temperature_drop = 10
nn_width = 2**4  # cannot be set in load
nn_depth = 4  # cannot be set in load
checkpoint_freq = 2
actors = 1
evaluators = 1
evaluation_window = 10
eval_levels = 4
max_steps = 0
quiet = False
verbose = True


def main(unused_argv):

    config = alpha_zero.Config(
        game=game,
        device=device,
        path=path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        l2_regularisation=l2_regularisation,
        momentum=momentum,
        train_batch_size=train_batch_size,
        replay_buffer_size=replay_buffer_size,
        replay_buffer_reuse=replay_buffer_reuse,
        max_steps=max_steps,
        checkpoint_freq=checkpoint_freq,
        actors=actors,
        evaluators=evaluators,
        uct_c=uct_c,
        max_simulations=max_simulations,
        policy_alpha=policy_alpha,
        policy_epsilon=policy_epsilon,
        temperature=temperature,
        temperature_drop=temperature_drop,
        evaluation_window=evaluation_window,
        eval_levels=eval_levels,
        nn_width=nn_width,
        nn_depth=nn_depth,
        observation_shape=None,
        output_size=None,
        quiet=quiet,
    )
    alpha_zero.alpha_zero(config, load=load)


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
