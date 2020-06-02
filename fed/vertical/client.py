#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/2, homeway'

import copy
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from utils import Model

class Client():
    def __init__(self, uid, conf, data_loader):
        self.uid = uid
        self.conf = conf
        self.data_loader = data_loader
        self.global_params = None
        self.init()

    def init(self):
        self.model = Model(self.conf.num_features, self.conf.num_classes)