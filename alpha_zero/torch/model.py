"""
This class is the wrapper to the neural net for calling from the training / inference.
Mostly will be adapted from https://github.com/jseppanen/azalea
Spcifically policy.py
"""

import os
from typing import Optional, Sequence
import logging
import torch
from torch import optim
from torch.optim import lr_scheduler

from alpha_zero.torch import network
from alpha_zero.torch.network import TrainInput


class Model(object):
    def __init__(self):
        """Construct a new model"""
        self.settings = {
            "move_sampling": False,
            "move_exploration": False,
        }
        pass

    @property
    def net(self):
        try:
            return self._net
        except AttributeError:
            raise RuntimeError("Policy must be initialized or loaded before use")

    @net.setter
    def net(self, net):
        self._net = net

    def build_model(self, config):
        """Build a model with specified params.
        IE create the model. This is the method you call at start of training."""
        print(config)
        device = torch.device(config["device"])
        self.device = device
        if device.type == "cuda":
            # enable cudnn auto-tuner
            torch.backends.cudnn.benchmark = True

        self.net = network.HexNetwork(
            input_shape=config["observation_shape"],
            output_size=config["output_size"],
            nn_depth=config["nn_depth"],
            nn_width=config["nn_width"],
        )

        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["l2_regularisation"],
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["learning_rate"],
            gamma=config["weight_decay"],
        )

        self.net.to(device)
        # don't train anything by default
        self.net.eval()

        self.input_shape = config["observation_shape"]
        self.output_size = config["output_size"]
        # network params
        self.nn_depth = config["nn_depth"]
        self.nn_width = config["nn_width"]
        self.path = config["path"]

        # search params
        self.lr = config["learning_rate"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.l2_regularisation = config["l2_regularisation"]
        self.search_batch_size = config["train_batch_size"]
        self.exploration_coef = config["uct_c"]
        self.exploration_depth = config["max_simulations"]
        self.exploration_noise_alpha = config["policy_alpha"]
        self.exploration_noise_scale = config["policy_epsilon"]
        self.exploration_temperature = config["temperature"]
        if "seed" in config:
            self.seed(config["seed"])

    def __del__(self):
        """Stop the model and free its memory"""

    def load_state_dict(self, state):
        """Load model state"""
        # load network architecture and params
        self.input_shape = state["observation_shape"]
        self.output_size = state["output_size"]
        self.nn_depth = state["nn_depth"]
        self.nn_width = state["nn_width"]
        self.net = network.HexNetwork(
            self.input_shape, self.output_size, self.nn_depth, self.nn_width
        )
        self.net.load_state_dict(state["net"])
        self.path = state["path"]

        # search params
        self.lr = state["learning_rate"]
        self.momentum = state["momentum"]
        self.weight_decay = state["weight_decay"]
        self.l2_regularisation = state["l2_regularisation"]
        self.search_batch_size = state["train_batch_size"]
        self.exploration_coef = state["uct_c"]
        self.exploration_depth = state["max_simulations"]
        self.exploration_noise_alpha = state["policy_alpha"]
        self.exploration_noise_scale = state["policy_epsilon"]
        self.exploration_temperature = state["temperature"]

    def num_trainable_variables(self):
        logging.info([p.shape for p in self.net.parameters()])
        return sum(p.numel() for p in self.net.parameters())

    def state_dict_new(self):
        """New version of the state dict, stores all non NN vars in a separate config."""
        return {"net": self.net.state_dict(), "config_path": os.path.join(self.path)}

    def state_dict(self):
        """Return model state
        Only serializes the (hyper)parameters, not ongoing game state (search tree etc)
        """
        return {
            "net": self.net.state_dict(),
            "path": self.path,
            "observation_shape": self.input_shape,
            "output_size": self.output_size,
            "nn_depth": self.nn_depth,
            "nn_width": self.nn_width,
            "learning_rate": self.lr,
            "momentum": self.momentum,
            "train_batch_size": self.search_batch_size,
            "uct_c": self.exploration_coef,
            "l2_regularisation": self.l2_regularisation,
            "weight_decay": self.weight_decay,
            "max_simulations": self.exploration_depth,
            "policy_alpha": self.exploration_noise_alpha,
            "policy_epsilon": self.exploration_noise_scale,
            "temperature": self.exploration_temperature,
        }

    def inference(self, observation, legals_mask, device="cpu"):
        """performs a step of inference on the model"""
        self.net.eval()
        with torch.set_grad_enabled(False):
            observation = torch.tensor(observation).to(device).float()
            legals_mask = torch.tensor(legals_mask).to(device).float()
            out = self.net.inference(observation, legals_mask)

        return out["value"], out["policy"]

    def update(self, batch, device="cpu"):
        """Runs a training step"""
        observation, legals_mask, policy, value = batch
        self.net.train()
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            observation = observation.to(device).float()
            legals_mask = legals_mask.to(device).float()
            policy = policy.to(device).float()
            value = value.to(device).float()

            # logging.info(f"{observation=}")
            # logging.info(f"{policy=}")
            # logging.info(f"{policy.shape=}")

            output, loss = self.net.update(observation, legals_mask, policy, value)
            loss = loss.float()
            # _, policy_loss, value_loss, l2_reg_loss = self._session.run(
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        return output, loss.item()

    def save_checkpoint(self, step, optimizer=True, replaybuf=None):
        """Save model (and replay buffer) checkpoint
        :param name: File base name
        :param replaybuf: Save also replay buffer
        """
        state = {
            "model": self.state_dict(),
        }
        if optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        path = os.path.join(self.path, f"checkpoint-{step}.policy.pth")
        torch.save(state, path)
        logging.info(f"saved policy checkpoint to {path}")

        # if replaybuf:
        #     rpath = f"checkpoint-{step}.replaybuf.pth"
        #     torch.save(replaybuf.state_dict(), rpath)
        #     logging.info(f"saved replay buffer checkpoint to {rpath}")
        return path

    def load_checkpoint(self, path: str):
        """Load a new model to an existing NN"""
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.net.eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "Model":
        """Create policy and load weights from checkpoint
        Paths can be local filenames or s3://... URL's (please install
        smart_open library for S3 support).
        Loads tensors according to device
        :param path: Either local or S3 path of policy file
        """
        policy = cls()
        if device:
            device = torch.device(device)
            location = device.type
            if location == "cuda":
                location += f":{device.index or 0}"
        else:
            location = None
        state = torch.load(path, map_location=location)
        policy.load_state_dict(state["policy"])
        policy.net.eval()
        if device:
            policy.net.to(device)
        return policy
