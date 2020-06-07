#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/5, homeway'

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, name, created_time):
        super(Model, self).__init__()
        self.name = name
        self.created_time = created_time

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return torch.nn.Parameter(Variable(torch.randn(*size) * xavier_stddev, requires_grad=True))

    def set_grad(self, grad_tensors):
        for i, param in enumerate(self.parameters()):
            param.grad = Variable(param.new().resize_as_(param).zero_())
            param.grad.data = grad_tensors[i]

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