#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import torch
import syft as sy
from utils.helper import Helper
#from data.uci import Data
from data.digist import Data

def run_horizontal(conf, helper):
    from fed.horizontal.client import Client
    from fed.horizontal.server import Server
    from fed.horizontal.adv_client import AdversaryClient
    from model.horizontal.credit import Credit as Model

    # syft initial
    conf.syft_clients = {}
    conf.syft_hook = sy.TorchHook(torch)
    for i in range(conf.num_clients):
        conf.syft_clients[i] = sy.VirtualWorker(conf.syft_hook, id=str(i))
    conf.syft_crypto_provider = sy.VirtualWorker(conf.syft_hook, id="james")

    # data initial
    data = Data(conf)
    data.load_data()

    # federated client
    for uid in range(conf.num_clients):
        if uid == 0:
            conf.fed_clients[uid] = AdversaryClient(uid, conf, data.data_loader[uid])
        else:
            conf.fed_clients[uid] = Client(uid, conf, data.data_loader[uid])

    # federated server
    server = Server(conf, data)
    server.run()

def run_vertical(conf, helper):
    '''Run vertical partition federated learning.
        Follow paper `Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption`
    '''
    from fed.vertical.client import Client
    from fed.vertical.arbiter import Arbiter

    # syft initial
    '''
    conf.num_clients = 2
    conf.syft_clients = {}
    conf.syft_hook = sy.TorchHook(torch)
    for i in range(conf.num_clients):
        conf.syft_clients[i] = sy.VirtualWorker(conf.syft_hook, id=str(i))
    conf.syft_crypto_provider = sy.VirtualWorker(conf.syft_hook, id="james")
    '''

    # data initial
    data = Data(conf)
    data.load_data(alpha=0.8)

    # federated client
    for uid, party in conf.fed_vertical["party"].items():
        conf.fed_clients[uid] = Client(uid, conf, data.data_loader[uid], party)

    # federated arbiter
    server = Arbiter(conf, data)
    server.run()

def main():
    # system initial
    helper = Helper()
    conf = helper.conf

    # horizontal/vertical
    if conf.fed_partition == "horizontal":
        run_horizontal(conf, helper)
    elif conf.fed_partition == "vertical":
        run_vertical(conf, helper)

if __name__ == "__main__":
    main()












