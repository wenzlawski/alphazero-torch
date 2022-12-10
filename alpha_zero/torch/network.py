"""
This class will contain all the nn.Models necessary of the AZ model. 
Mostly will be adapted from https://github.com/jseppanen/azale
"""

from typing import Sequence
import collections
import logging

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def conv3x3(in_chans, out_chans) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)


def conv1x1(in_chans, out_chans) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)


class TrainInput(
    collections.namedtuple("TrainInput", "observation legals_mask policy value")
):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        observation, legals_mask, policy, value = zip(*train_inputs)
        return TrainInput(
            np.array(observation, dtype=np.float32),
            np.array(legals_mask, dtype=np.bool),
            np.array(policy),
            np.expand_dims(value, 1),
        )


class Losses(collections.namedtuple("Losses", "policy value l2")):
    """Losses from a training step."""

    @property
    def total(self):
        return self.policy + self.value + self.l2

    def __str__(self):
        return (
            "Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, " "l2: {:.3f})"
        ).format(self.total, self.policy, self.value, self.l2)

    def __add__(self, other):
        return Losses(
            self.policy + other.policy, self.value + other.value, self.l2 + other.l2
        )

    def __truediv__(self, n: int):
        return Losses(self.policy / n, self.value / n, self.l2 / n)


# ! EXPLAIN or is there a newer version of this?
# Corresponds to Resudiual net, with the number of Resblocks equal to NN depth
class Resblock(nn.Module):
    def __init__(self, in_dim, dim) -> None:
        super().__init__()
        if dim != in_dim:
            self.res_conv = conv1x1(in_dim, dim)
            self.res_bn = nn.BatchNorm2d(dim)
        else:
            self.res_conv = self.res_bn = None

        self.conv1 = conv3x3(in_dim, dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # residual connection
        if self.res_conv:
            identity = self.res_bn(self.res_conv(x))
        out += identity
        out = self.relu(out)
        return out


class Network(nn.Module):
    def __init__(
        self, input_shape, output_size, nn_depth, nn_width, value_chans, policy_chans
    ):
        super().__init__()

        self.input_shape = input_shape

        # Input upsampling
        self.conv1 = conv3x3(input_shape[0], nn_width)
        self.bn1 = nn.BatchNorm2d(nn_width)

        # Initiate residual blocks
        blocks = [Resblock(nn_width, nn_width) for i in range(nn_depth)]
        self.resblocks = nn.Sequential(*blocks)

        # Construct the value head
        self.value_conv1 = conv1x1(nn_width, value_chans)
        self.value_bn1 = nn.BatchNorm2d(value_chans)
        self.value_fc2 = nn.Linear(value_chans * output_size, nn_width)
        self.value_fc3 = nn.Linear(nn_width, 1)

        # Construct the policy head
        self.move_conv1 = conv1x1(nn_width, policy_chans)
        self.move_bn1 = nn.BatchNorm2d(policy_chans)

        self.relu = nn.ReLU(inplace=True)

    @property
    def device(self):
        """get current device of model."""
        return self.conv1.weight.device

    def forward(self, x):
        """
        ???:param x: Batch of game boards (batch x height x width, int32)
        """

        x = x.to(self.conv1.weight.dtype)

        out = x.view((-1, *self.input_shape))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply residual blocks
        out = self.resblocks(out)

        # Compute through the value head
        v = self.value_conv1(out)
        v = self.value_bn1(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc2(v)
        v = self.relu(v)
        v = self.value_fc3(v)
        value = torch.tanh(v).squeeze(1)

        # Compute policy head
        p = self.move_conv1(out)
        p = self.move_bn1(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)

        return value, p

    def update(self, observation, legals_mask, policy, value):
        """
        Runs a training step.
        """
        output = self.forward(observation, legals_mask)

        # Eq. (1) without regularization (done with weight_decay)
        value_loss = F.mse_loss(output["value"], value)
        moves_loss = -(policy * torch.squeeze(output["policy"])).sum() / len(policy)
        loss = value_loss.to(moves_loss.device) + moves_loss
        # loss = loss.float()
        output = dict((k, v.detach()) for k, v in output.items())
        output.update(value_loss=value_loss.item(), moves_loss=moves_loss.item())
        return output, loss

    def inference(self, observation, legals_mask):
        """Performes inference on the model."""
        # ! TAKEN from the aza model
        output = self.forward(observation, legals_mask)
        output = dict((k, v.detach()) for k, v in output.items())
        return output

    def load(self, modelpath):
        state = torch.load(modelpath)
        self.load_state_dict(state["model"])
        return state["optimizer"]

    def save(self, modelpath, optimizer):
        state = {"model": self.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, modelpath)


class HexNetwork(Network):
    """An AlphaZero style model with a policy and value head.

    This model is the interface for model.py.

    """

    def __init__(self, input_shape, output_size, nn_depth, nn_width):

        # ! Watch out for input dim -> what is it in the OS model?
        value_chans = 1
        policy_chans = 2
        super().__init__(
            input_shape=input_shape,
            output_size=output_size,
            nn_depth=nn_depth,
            nn_width=nn_width,
            value_chans=value_chans,
            policy_chans=policy_chans,
        )

        # policy head
        self.move_fc = nn.Linear(policy_chans * output_size, output_size)

    def forward(self, x, legal_moves):
        """
        legal_moves padded with zeros
        :param x: Batch of game boards (batch x height x width, int32)
        :param legal_moves: Batch of legal moves (batch x MAX_MOVES, int32)
        ? How are legal moves handeled here?
        """

        # resnet

        value, p = super().forward(x)
        # policy head
        moves_logit = self.move_fc(p)
        legal_tiles = legal_moves.clamp(min=0)
        moves_logit = torch.gather(moves_logit, 1, legal_tiles.long())
        # clear padding
        moves_logit.masked_fill_(legal_moves == 0, -99)
        policy = F.log_softmax(moves_logit, dim=1)
        return dict(value=value, policy=policy)
