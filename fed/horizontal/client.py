#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import copy
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from model.horizontal.credit import Credit as Model

class Client():
    def __init__(self, uid, conf, data_loader):
        self.uid = uid
        self.conf = conf
        self.data_loader = data_loader
        self.global_params = None
        self.init()

    def init(self):
        self.model = Model(self.conf.num_features, self.conf.num_classes).to(self.conf.device)

    def update(self, parameters):
        self.global_params = copy.deepcopy(parameters)
        self.model.copy_params(self.global_params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.conf.fed_learning_rate)

        self.model.train()
        for e in range(self.conf.fed_epoch):
            train_loss, train_acc = 0, 0
            for step, (x, y) in enumerate(self.data_loader):
                x, y = x.to(self.conf.device), y.to(self.conf.device)
                self.optimizer.zero_grad()
                logists = self.model(x)
                loss = F.cross_entropy(logists, y)
                loss.backward()

                pred = logists.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(y.view_as(pred)).sum().item()
                train_loss += F.cross_entropy(logists, y, reduction='sum').item()
                self.optimizer.step()
        print("-> client{:d} finish! train_loss={:.3f} train_acc={:.1f}%".format(
            self.uid,
            train_loss / len(self.data_loader.dataset),
            100 * train_acc / len(self.data_loader.dataset))
        )

        if self.conf.fed_horizontal["encrypt_weight"]:
            return self.encrypted_model(self.model)
        return self.model.state_dict()

    def encrypted_model(self, model):
        spdz_weights = OrderedDict()
        for name, params in model.named_parameters():
            # select the identical parameter from each worker and copy it
            copy_of_parameter = (params.data - self.global_params[name]).copy()

            # since SMPC can only work with integers (not floats), we need
            # to use Integers to store decimal information. In other words,
            # we need to use "Fixed Precision" encoding.
            fixed_precision_param = copy_of_parameter.fix_precision()

            # now we encrypt it on the remote machine. Note that
            # fixed_precision_param is ALREADY a pointer. Thus, when
            # we call share, it actually encrypts the data that the
            # data is pointing TO. This returns a POINTER to the
            # MPC secret shared object, which we need to fetch.
            encrypted_param = fixed_precision_param.share(*self.conf.syft_clients.values(), crypto_provider=self.conf.syft_crypto_provider)

            # now we fetch the pointer to the MPC shared value
            param = encrypted_param.get()

            # save the parameter so we can average it with the same parameter
            # from the other workers
            spdz_weights[name] = param
        return spdz_weights
















