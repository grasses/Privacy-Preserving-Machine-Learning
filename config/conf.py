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
    num_round = 20
    num_clients = 10
    num_per_round = 10
    fed_epoch = 10
    fed_learning_rate = 0.0005
    fed_clients = {}
    fed_aggregate = "avg"
    fed_partition = "horizontal"    # data partition: horizontal/vertical
    fed_horizontal = {

    }
    fed_vertical = {
        "split": 10,
    }

    # syft
    syft_hook = None
    syft_clients = {}
    syft_crypto_provider = None