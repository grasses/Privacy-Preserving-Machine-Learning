#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2019/11/22, homeway'

import torch, torch.nn as nn, torch.nn.functional as F
from collections import OrderedDict
from model import Model

class Credit(Model):
    def __init__(self, num_features, num_classes, name="", created_time=""):
        super(Credit, self).__init__(name, created_time)
        self.linear_blocks = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.Dropout2d(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.linear_blocks(x)
        x = torch.sigmoid(x)
        return x