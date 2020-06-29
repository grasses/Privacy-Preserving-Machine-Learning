#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import os
import random
import torch
import numpy as np
import datetime

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
format_time = str(datetime.datetime.now().strftime('%m%d%H%M%S'))


class Party():
    def __init__(self, uid, num_feature=10, num_output=1):
        self.uid = uid
        self.num_features = num_feature
        self.num_output = num_output
        self.public_key = None
        self.secret_key = None


class Conf():
    # machine learning
    batch_size = 1
    momentum = 0.8
    learning_rate = 0.1
    num_features = 64
    num_classes = 10

    # federated learning
    num_round = 10000
    num_clients = 2
    num_per_round = 2
    fed_clients = {}
    fed_aggregate = "avg"
    fed_partition = "vertical"  # data partition: horizontal/vertical
    fed_horizontal = {
        "encrypt_weight": False,
        "local_epoch": 20
    }
    fed_vertical = {
        "party": {
            0: Party(0, 30, num_classes),
            1: Party(1, 34, num_classes)
        },
        "split": 30,
        "num_steps": -1,        # update in dataloader
        "num_attack": 300,      # iteration of attack
        "attack_steps": [1]     # selected attack steps
    }

    # syft
    syft_hook = None
    syft_clients = {}
    syft_crypto_provider = None

    # system
    logger = "warn"
    dataset = "digist"
    scope_name = f"{dataset}_{fed_partition}_{format_time}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






