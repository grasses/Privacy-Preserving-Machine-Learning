
#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/5, homeway'

import torch, torch.nn as nn, torch.nn.functional as F
from collections import OrderedDict
from model import Model

class Generator(Model):
    def __init__(self, num_input, num_output, name="", created_time=""):
        super(Generator, self).__init__(name, created_time)
        self.linear_blocks = nn.Sequential(
            nn.Linear(num_input, 32),
            nn.Linear(32, num_output)
        )

    def forward(self, z, c):
        data = torch.cat([z, c], 1)
        data = self.linear_blocks(data)
        return data