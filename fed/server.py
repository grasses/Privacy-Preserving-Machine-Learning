#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import time
import copy
import numpy as np
import syft as sy
from utils.helper import Helper


class Server():
    def __init__(self, conf, data, model):
        self.conf = conf
        self.helper = Helper(conf)
        self.data = data
        self.model = model
        self.init()

    def init(self):
        print("-> init server")
        self.fed_clients = self.conf.fed_clients
        self.syft_clients = self.conf.syft_clients

    def run(self):
        # training process
        for t in range(self.conf.num_round):
            start = time.time()

            # random select client
            curr_client_ids = np.random.choice(self.conf.num_clients, self.conf.per_round, replace=False)
            print(f"\n\n-> t={t} agent_list{curr_client_ids}")

            # client update
            spdz_weights = {}
            for uid in curr_client_ids:
                spdz_weights[uid] = self.fed_clients[uid].update(copy.deepcopy(self.model))
                new_weights = self.aggregation(spdz_weights)
                self.model.copy_params(new_weights)

            stop = time.time()
            print("-> end round:{} using:{} s".format(t, int(start-stop)))

    def aggregation(self, spdz_weights):
        if self.conf.fed_aggregate == "avg":
            return self.helper.weights_avg(self.model.state_dict(), spdz_weights)












