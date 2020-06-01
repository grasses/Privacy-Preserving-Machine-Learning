#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class Client():
    def __init__(self, uid, conf, data_loader):
        self.uid = uid
        self.conf = conf
        self.data_loader = data_loader
        self.model = None

    def update(self, model):
        print("-> client{} update".format(self.uid))
        self.model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.conf.fed_learning_rate)
        for e in range(self.conf.fed_epoch):
            for step, (x, y) in enumerate(self.data_loader):
                self.model.send(self.uid)
                self.optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                self.optimizer.step()
        return self.encrypted_model(self.model)

    def encrypted_model(self, model):
        spdz_weights = OrderedDict()
        for name, params in model.named_parameters():
            # select the identical parameter from each worker and copy it
            copy_of_parameter = params.copy()

            # since SMPC can only work with integers (not floats), we need
            # to use Integers to store decimal information. In other words,
            # we need to use "Fixed Precision" encoding.
            fixed_precision_param = copy_of_parameter.fix_precision()

            # now we encrypt it on the remote machine. Note that
            # fixed_precision_param is ALREADY a pointer. Thus, when
            # we call share, it actually encrypts the data that the
            # data is pointing TO. This returns a POINTER to the
            # MPC secret shared object, which we need to fetch.
            encrypted_param = fixed_precision_param.share(*self.conf.syft_clients, crypto_provider=self.conf.syft_crypto_provider)

            # now we fetch the pointer to the MPC shared value
            param = encrypted_param.get()

            # save the parameter so we can average it with the same parameter
            # from the other workers
            spdz_weights[name] = param
        return spdz_weights
















