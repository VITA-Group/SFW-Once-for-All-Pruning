from abc import ABC, abstractmethod
from os import path as osp
import torch
from torch.nn import Module
import torch.nn.functional as F


class BaseNetwork(Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    def get_snapshot(self, key_must_have=''):
        new_state_dict = {}
        state_dict = self.state_dict()
        if key_must_have == '':
            new_state_dict = state_dict
        else:
            for k, v in state_dict.items():
                if key_must_have in k:
                    new_state_dict[k] = v
        return new_state_dict

    def load_snapshot(self, loaded_state_dict, key_must_have=''):
        state_dict = self.state_dict()
        if key_must_have == '':
            state_dict = loaded_state_dict
        else:
            for k, v in loaded_state_dict.items():
                if key_must_have in k:
                    state_dict[k] = v
        self.load_state_dict(state_dict)

    def has_nan(self):
        for parameter in self.parameters():
            if torch.any(torch.isnan(parameter)):
                return True
        return False


def get_nonlinearity(act_name='relu'):
    nonlinearity_dict = {
        'relu': F.relu,
        # 'swish': swish,
        'tanh': torch.tanh,
        'identity': lambda x: x,
    }
    return nonlinearity_dict[act_name]