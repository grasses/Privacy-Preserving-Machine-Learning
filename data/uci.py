#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import os
import numpy as np
from utils.data import split_data
from torch.utils.data import Dataset, DataLoader

class UCIDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class Data():
    def __init__(self, conf, user_list=[]):
        self.conf = conf
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_x = None
        self.splited_data = {}
        self.data_loader = {}
        self.test_loader = None
        self.user_list = user_list

    def load_data(self, sep=0.8):
        raw_data = np.genfromtxt(os.path.join(self.conf.data_path, "index.csv"), delimiter=",")
        data = np.delete(raw_data, 0, axis=1)

        size = len(data)
        train_size = int(size * sep)
        idx = np.random.choice(size, size, replace=False)
        data = data[idx]

        # split data into train & test
        self.train_x = data[:train_size, :-1]
        self.train_y = data[:train_size, -1]
        self.test_x = data[train_size:, :-1]
        self.test_y = data[train_size:, -1]

        # split data from users
        self.splited_data = split_data(self.train_x, self.train_y, num_clients=self.conf.num_clients,
                                       num_classes=self.conf.num_classes)

        # build data loader & send to user
        for uid, data in self.splited_data.items():
            self.data_loader[uid] = DataLoader(UCIDataset(data[0], data[1]), batch_size=self.conf.batch_size, shuffle=True)

            print("-> send data to:{}, size:{}".format(uid, len(data[0])))
            for batch_idx, (data, target) in enumerate(self.data_loader[uid]):
                data.send(self.conf.syft_clients[uid])
                target.send(self.conf.syft_clients[uid])

        # build test loader
        self.test_loader = DataLoader(UCIDataset(self.test_x, self.test_y), batch_size=self.conf.batch_size, shuffle=False)