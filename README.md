# AI and Games Project

Software required to carry out the project to make AI for the game of Hex.

## Training the AlphaZero model

### Parameters

### Resuming the training

The problem we are facing is that the model does not have a 'resume training from checkpoint' function.
When you start training after stopping it, it will overwrite all previous files.
This is obviously bad, because we want to train the model for free, which means
resuming training over a numer of different google cloud accounts with free credits.

What we need to do is to write a script `resume_alpha_zero.py` which takes a folder
of a previous training session, reads the parameters from `config.json` and instaniates
the actors, evaluators, and learner from the checkpoint.

## Playing with the model

Because we want to let our model play on the tournament server in the end, 
we need to write our own implementation of the algorithm in pytorch, and connet
it with the API for the tournament machine, which cam be found in `agents/DefaultAgents/NaiveAgent.py`.


### Interfacing with the tournament machine

As a starting point we can take the 'play against the agent' from the Openspiel documentation:

```python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=az --az_path <path to your checkpoint directory>```

This should give us pointers as to what functionality we need to provide in order to let our agent 
interface with a generic API.

### Converting the checkpoint to a Pytorch model weight