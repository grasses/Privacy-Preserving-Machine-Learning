#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import os
import torch

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
    batch_size = 1000
    learning_rate = 0.001

    # federated learning
    num_round = 20
    num_clients = 10
    num_classes = 2
    per_round = 10
    fed_epoch = 10
    fed_learning_rate = 0.001
    fed_clients = {}
    fed_aggregate = "avg"

    # syft
    syft_hook = None
    syft_clients = {}
    syft_crypto_provider = {}