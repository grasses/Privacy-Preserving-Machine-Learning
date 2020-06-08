#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import os
import random
import torch
random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

class Party():
    def __init__(self, uid, num_feature=10, num_output=1):
        self.uid = uid
        self.num_features = num_feature
        self.num_output = num_output
        self.public_key = None
        self.secret_key = None

class Conf():
    ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = "uci_credit"
    scope_name = ""
    data_root = os.path.join(ROOT, "data")
    data_path = os.path.join(data_root, dataset)
    model_path = os.path.join(ROOT, "model")
    output_path = os.path.join(ROOT, "output")

    # machine learning
    batch_size = 200
    learning_rate = 0.001
    num_features = 23
    num_classes = 2

    # federated learning
    num_round = 20000
    num_clients = 2
    num_per_round = 2
    fed_epoch = 10
    fed_learning_rate = 0.02
    fed_clients = {}
    fed_aggregate = "avg"
    fed_partition = "horizontal"  # data partition: horizontal/vertical
    fed_horizontal = {
        "encrypt_weight": False
    }
    fed_vertical = {
        "party": {
            0: Party(0, 13, 1),
            1: Party(1, 10, 1)
        },
        "split": 10,
        "num_steps": 0,
    }

    # syft
    syft_hook = None
    syft_clients = {}
    syft_crypto_provider = None






