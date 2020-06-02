#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/3, homeway'

import time
import copy
import numpy as np
from utils.helper import Helper
from utils.eval import Evaluation


class Server():
    def __init__(self, conf, data, model):
        self.conf = conf
        self.data = data
        self.model = model
        self.fed_clients = self.conf.fed_clients
        self.syft_clients = self.conf.syft_clients
        self.init()

    def init(self):
        print("-> initial server")
        self.helper = Helper(self.conf)
        self.eval = Evaluation(self.conf, self.data)

        # creat homomorphic encryption key pair


    def run(self):
        pass