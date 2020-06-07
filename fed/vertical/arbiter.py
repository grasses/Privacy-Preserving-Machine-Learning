#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/3, homeway'

import time
import torch
import torch.optim as optim
from utils.helper import Helper
from model.vertical.credit import PartyModel as Model

class Arbiter():
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.model = {}
        self.optimizer = {}
        self.fed_clients = self.conf.fed_clients
        self.syft_clients = self.conf.syft_clients
        self.party = self.conf.fed_vertical["party"]
        self.init()

    def init(self):
        print("-> initial server")
        self.helper = Helper(self.conf)

        # creat homomorphic encryption key pair
        # TODO

        # setup model
        for uid, party in self.party.items():
            self.model[uid] = Model(party.num_features, party.num_output).to(self.conf.device)
            self.optimizer[uid] = optim.SGD(self.model[uid].parameters(), lr=self.conf.learning_rate, momentum=0.9)

    def run(self):
        for step in range(self.conf.num_round * self.conf.fed_vertical["num_steps"]):
            # start round for all parties
            for uid, party in self.party.items():
                parameters = self.model[uid].state_dict()
                self.fed_clients[uid].start_round(parameters)

            # three steps for grad
            u_prime = self.fed_clients[0].grad_step1()
            w, z = self.fed_clients[1].grad_step2(u_prime)
            grad = self.fed_clients[0].grad_step3(w, z)

            # update model
            for uid, party in self.party.items():
                self.model[uid].set_grad([grad[uid]])
                self.optimizer[uid].step()

            # run a batch forward
            if step % 20 == 0:
                logists = {}
                for uid, party in self.party.items():
                    logists[uid] = self.fed_clients[uid].forward().float()
                logists = logists[0] + logists[1]
                self.fed_clients[0].fetch_evaluation(logists, step)

            if step == 10000:
                exit(1)


























