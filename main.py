#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'


import torch
import syft as sy
from utils.helper import Helper
from data.uci import Data
from utils.model import Credit as Model
from fed.client import Client
from fed.server import Server

def main():
    # system initial
    helper = Helper()
    conf = helper.conf

    # syft initial
    conf.syft_hook = sy.TorchHook(torch)
    conf.syft_clients = {}
    for i in range(conf.num_clients):
        conf.syft_clients[i] = sy.VirtualWorker(conf.syft_hook, id=str(i))
    conf.syft_crypto_provider = sy.VirtualWorker(conf.syft_hook, id="james")

    # data initial
    data = Data(conf)
    data.load_data()

    # federated client
    for uid in range(conf.num_clients):
        conf.fed_clients[uid] = Client(uid, conf, data.data_loader[i])

    # federated server
    server = Server(conf, data, Model(num_features=conf.num_features, num_classes=conf.num_classes))
    server.run()

if __name__ == "__main__":
    main()