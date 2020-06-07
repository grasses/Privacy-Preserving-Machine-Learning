#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import time
import copy
import numpy as np
from utils.helper import Helper
from utils.eval import Evaluation
from model.horizontal import Model

class Server():
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.fed_clients = self.conf.fed_clients
        self.syft_clients = self.conf.syft_clients
        self.init()

    def init(self):
        print("-> initial server")
        self.helper = Helper(self.conf)
        self.eval = Evaluation(self.conf, self.data)
        self.model = Model(num_features=self.conf.num_features, num_classes=self.conf.num_classes).to(self.conf.device)

    def run(self):
        # training process
        for t in range(self.conf.num_round):
            start = time.time()

            # random select client
            curr_client_ids = np.random.choice(self.conf.num_clients, self.conf.num_per_round, replace=False)
            print(f"\n\n-> t={t} selected_client={curr_client_ids}")

            # client update
            spdz_weights = {}
            for uid in curr_client_ids:
                spdz_weights[uid] = self.fed_clients[uid].update(self.model.state_dict())

            if not self.conf.fed_horizontal["encrypt_weight"]:
                print("-> client send params to server without shared_encrypt")

            new_weights = self.aggregation(spdz_weights)
            self.model.copy_params(new_weights)

            stop = time.time()
            print("-> end round:{:d} using:{:.2f}s".format(t, float(stop-start)))

            # evluation
            self.eval.eval_test(self.model)

    def aggregation(self, spdz_weights):
        if len(spdz_weights) == 1:
            return spdz_weights[0]

        if self.conf.fed_aggregate == "avg":
            return self.helper.weights_avg(self.model.state_dict(), spdz_weights,
                                           fix_precision=self.conf.fed_horizontal["encrypt_weight"])












