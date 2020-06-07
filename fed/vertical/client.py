#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/2, homeway'

import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from model.vertical.credit import PartyModel

class Client():
    def __init__(self, uid, conf, data_loader, party, debug=False):
        self.uid = uid
        self.conf = conf
        self.party = party
        self.data_point = -1
        self.data_loader = data_loader
        self.global_params = None
        self.public_key = None
        self.debug = debug
        self.init()

    def init(self):
        self.model = PartyModel(self.party.num_features, self.party.num_output).to(self.conf.device)
        print("-> party={} w={}".format(vars(self.party), self.model.w.shape))

    def load_batch(self, point):
        x, y = list(self.data_loader)[point]
        x, y = x.to(self.conf.device), y.to(self.conf.device)

        #if self.uid == 0:
            #print("-> label={} shape={}".format(y.view(-1), y.shape))

        return x, y

    def start_round(self, parameters, mask=None, public_key=None):
        self.global_params = copy.deepcopy(parameters)
        self.model.copy_params(self.global_params)

        # TODO: use mask to load batch data
        self.data_point += 1
        self.data_point = self.data_point % len(self.data_loader)
        self.x, self.y = self.load_batch(self.data_point)

        if public_key is not None:
            self.public_key = public_key

    def stop_round(self):
        pass

    def grad_step1(self):
        u_prime = self.model.step1(self.x, self.y)
        return u_prime

    def grad_step2(self, u_prime):
        w, z = self.model.step2(self.x, u_prime)
        return w, z

    def grad_step3(self, w, z):
        z_prime = self.model.step3(self.x, w)
        return z_prime.t(), z.t()

    def forward(self, parameters=None):
        if parameters is not None:
            self.global_params = copy.deepcopy(parameters)
            self.model.copy_params(self.global_params)
        return self.model(self.x)

    def fetch_evaluation(self, logists, step=0):
        label = copy.deepcopy(self.y).view(-1)
        pred = logists.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(label.view_as(pred)).sum().item()
        loss = torch.nn.MSELoss()(logists, label).item()

        print("-> step={:d} train_loss={:.3f} train_acc={:.1f}%".format(
                step,
                loss / len(label),
                100 * accuracy / len(label)
            )
        )




































