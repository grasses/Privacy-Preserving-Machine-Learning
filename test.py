#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/19, homeway'

import torch
from utils.helper import Helper

class Data():
    def __init__(self):
        self.data_loader = {
            0: [
                [
                    torch.tensor([
                        [4.0], [1.0]
                    ]), torch.tensor([
                        [1], [-1]
                    ])
                ]
            ],
            1: [
                [
                    torch.tensor([
                        [0.0, 2.0, 3.0], [5.0, 3.0, 2.0]
                    ]), torch.tensor([
                        [0], [0]
                    ])
                ]
            ]
        }

def test_algorithm3(conf, helper):
    from fed.vertical.client import Client
    from fed.vertical.arbiter import Arbiter

    # init data
    data = Data()

    # federated client
    for uid, party in conf.fed_vertical["party"].items():
        conf.fed_clients[uid] = Client(uid, conf, data.data_loader[uid], party)

    # federated arbiter
    w1 = torch.tensor(
        [[0.0]], requires_grad=True
    )
    w2 = torch.tensor(
        [[2.0], [3.0], [3.0]], requires_grad=True
    )
    server = Arbiter(conf, data, w1, w2)
    server.run()


if __name__ == "__main__":
    # system initial
    helper = Helper()
    conf = helper.conf
    test_algorithm3(conf, helper)


















