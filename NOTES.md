# Notes on the individual implementiations of AZ for Hex

The ultimate goal to keep in mind is to build a alphazero version in pytorch modeled
on the training and interface code of openspiel.

This means:

training:
- parallelised self play
- multiple evaluators and actors
- gpu support
- checkpoint creation
- training the model from scratch
- loading and continuing training from a checkpiont


**Neurohex**

Uses Q-learning -> not useful for our application.

**Azalea**

- Alpha zero implementation of the Hex board game -> jackpot
- Has very good implementation of the model.
- Parallelised self play, training on a GPU
- neg: on pytorch version 0.4
- only single GPU
- using 6 blocks with width of 64 -> substantially smaller.
- Try 8 blocks with width 128 on for size
- Or 12 bolcks of width 64
- ! we can use the model and part of the training loop.
- improve training by randomly reflecting a subset of the batch 

**policy_trainer.py/train** is the training loop

**policy.py** is the wrapper class for the MCTS / aZ

in actuality we don't need to modify the training loop one bit.

We only have to implement an equivalent model wrapper and model.
And make it parralelisable on multiple GPUs.

- need to figure out the input formats to the model from OS
- figure the input formats that azalea takes


To figure out the input formats of openspiel, we can just start an actor and let
it play a couple of games.

Also check the model inputs


Note from openspiel: look at the replay buffer and the cache. 
These will potentially be lost when stopping the model training.
maybe pickle them together with the checkpoint functions.

## On the NNet

input size is gs.observation_tensor_shape()  -> [9, 11, 11]

Why 9 dim?

output size is gs.num_distinct_actions() -> 121
