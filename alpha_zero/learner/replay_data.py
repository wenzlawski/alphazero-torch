import torch
from torch.utils.data import Dataset
import numpy as np


class ReplayData(Dataset):
    def __init__(self, data):
        observation, legals_mask, policy, value = data
        self.observations = observation.astype(float)
        self.legals_mask = legals_mask.astype(float)
        self.policy = policy.astype(float)
        self.value = value.astype(float)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observations = self.observations[idx]
        legals_mask = self.legals_mask[idx]
        policy = self.policy[idx]
        value = self.value[idx]

        return observations, legals_mask, policy, value
