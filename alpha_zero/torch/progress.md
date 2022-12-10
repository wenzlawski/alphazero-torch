We need to convert the alphazero model and training loop to pytorch.

For that following things need be done
- model.py needs to be converted to a proper nn.Module
- modify the load from checkpoint and build graph methods
- need to modify the model.py to use relative sizes, passed from the creation.
- alpha_zero.py : 







Resources:
- https://github.com/geochri/AlphaZero_Chess/tree/master/src for the model
- https://github.com/jseppanen/azalea for the model also 