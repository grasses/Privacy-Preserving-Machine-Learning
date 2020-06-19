#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/30, homeway'

import os
import torch
import copy
import numpy as np
from utils.data import split_data
from torch.utils.data import Dataset, DataLoader

class UCIDataset(Dataset):
    def __init__(self, x, y, transform=None, worker=None):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform
        self.worker = worker
        if worker is not None:
            self.send()

    def send(self):
        self.x = self.x.send(self.worker)
        self.y = self.y.send(self.worker)

    def __getitem__(self, index):
        y = self.y[index]
        x = self.x[index] if self.transform is None else self.transform(self.x[index])
        return x, y

    def __len__(self):
        return len(self.x)

class Data():
    def __init__(self, conf):
        self.conf = conf
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_x = None
        self.splited_data = {}
        self.data_loader = {}
        self.test_loader = None

    def split_horizontal_data(self, num_clients, num_classes, batch_size=200):
        # split data from users
        self.splited_data = split_data(self.train_x, self.train_y, num_clients, num_classes)

        # build data loader & send to user
        for uid, item in self.splited_data.items():
            size = len(item[0])
            idx = np.random.choice(size, size, replace=False)
            item[0] = item[0][idx]
            item[1] = item[1][idx]

            y = item[1]
            z = len(np.where(y == 0)[0])
            zz = len(np.where(y == 1)[0])
            print(f"-> uid={uid} 0={z}, 1={zz} totoal={item[1].shape}\ny={y[:500]}")
            print("-> send data to client:{}, size:{}".format(uid, len(item[1])))
            dataset = UCIDataset(item[0], item[1])
            self.data_loader[uid] = DataLoader(dataset, batch_size, shuffle=True)

        # build test loader
        dataset = UCIDataset(self.test_x, self.test_y)
        self.test_loader = DataLoader(dataset, batch_size, shuffle=False)

    def split_vertical_data(self, args, batch_size=200):
        train_x_A = copy.deepcopy(self.train_x[:, :args["split"]])
        train_x_B = copy.deepcopy(self.train_x[:, args["split"]:])

        train_y_A = np.reshape(self.train_y, [-1, 1])
        train_y_B = np.zeros(self.train_y.shape)
        self.splited_data = {
            0: (train_x_A, train_y_A),
            1: (train_x_B, train_y_B)
        }

        test_x_A = self.test_x[:, :args["split"]]
        test_x_B = self.test_x[:, args["split"]:]
        test_y_A = self.test_y
        test_y_B = np.zeros(self.test_y.shape)
        self.splited_test_data = {
            0: (test_x_A, test_y_A),
            1: (test_x_B, test_y_B)
        }

        # TODO:PSI
        # build data loader & send to user
        for uid, item in self.splited_data.items():
            print("-> send data to client:{}, size:{}".format(uid, len(item[1])))
            dataset = UCIDataset(item[0], item[1])
            self.data_loader[uid] = DataLoader(dataset, batch_size, shuffle=False) # keep False
            # update num_steps/per round
            self.conf.fed_vertical["num_steps"] = len(self.data_loader[uid])

        # build test loader
        self.test_loader = {}
        for uid, item in self.splited_test_data.items():
            print("-> send testing data to client:{}, size:{}".format(uid, len(item[1])))

            dataset = UCIDataset(item[0], item[1])
            self.test_loader[uid] = DataLoader(dataset, batch_size, shuffle=False)

    def transform(self, x):
        for i in range(len(x[0])):
            if abs(x[0, i]) > 4:
                x_min = np.min(x[:, i])
                x_max = np.max(x[:, i])
                dist = (x_max - x_min)
                for j in range(len(x)):
                    x[j][i] = (x[j][i] - x_min) / dist

    def load_data(self, factor=0.8):
        raw_data = np.genfromtxt(os.path.join(self.conf.data_path, "index.csv"), delimiter=",")
        data = np.delete(raw_data, 0, axis=1)
        size = len(data)
        train_size = int(size * factor)
        idx = np.random.choice(size, size, replace=False)
        data = data[idx]

        # split data into train & test
        self.train_x = data[:train_size, :-1]
        self.train_y = data[:train_size, -1]
        self.test_x = data[train_size:, :-1]
        self.test_y = data[train_size:, -1]

        # horizontal/vertical
        if self.conf.fed_partition == "horizontal":
            self.split_horizontal_data(self.conf.num_clients, self.conf.num_classes, self.conf.batch_size)
        elif self.conf.fed_partition == "vertical":
            print("-> vertically split data")
            # according to paper: y = -1/1
            idx = np.where(self.train_y == 0)
            self.train_y[idx] = -1
            idx = np.where(self.test_y == 0)
            self.test_y[idx] = -1

            self.transform(self.train_x)
            self.transform(self.test_x)
            self.split_vertical_data(self.conf.fed_vertical, self.conf.batch_size)















