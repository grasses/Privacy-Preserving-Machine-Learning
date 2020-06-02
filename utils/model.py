#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2019/11/22, homeway'

import torch, torch.nn as nn, torch.nn.functional as F
from collections import OrderedDict

class ModelBase(nn.Module):
    def __init__(self, name, created_time):
        super(ModelBase, self).__init__()
        self.name = name
        self.created_time = created_time

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)

    def get_parameters(self, tid):
        for name, params in self.named_parameters():
            if name[:6] == "linear":
                if tid in name:
                    yield params
            else:
                yield params

class Credit(ModelBase):
    def __init__(self, num_features, num_classes, name="", created_time=""):
        super(Credit, self).__init__(name, created_time)
        self.linear_blocks = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Dropout2d(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.linear_blocks(x)
        x = torch.sigmoid(x)
        return x