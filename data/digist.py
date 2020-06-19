#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/15, homeway'

from sklearn import datasets
import copy
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataSet, DataLoader
from utils.data import split_data
from config.conf import Conf


class Dataset(TorchDataSet):
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
    def __init__(self, conf=Conf):
        self.conf = conf
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_x = None
        self.splited_data = {}
        self.data_loader = {}
        self.test_loader = None

    def split_horizontal_data(self):
        pass

    def split_vertical_data(self, args, batch_size=200):
        # convert to one hot
        self.test_y = (np.eye(10)[self.test_y]) * 2 - 1
        self.train_y = (np.eye(10)[self.train_y]) * 2 - 1

        # reshape x to vector
        self.test_x = np.reshape(self.test_x, [-1, 64])
        self.train_x = np.reshape(self.train_x, [-1, 64])

        # vertically split dataset
        train_x_A = copy.deepcopy(self.train_x[:, :args["split"]])
        train_x_B = copy.deepcopy(self.train_x[:, args["split"]:])
        train_y_A = np.reshape(self.train_y, [-1, 10])
        train_y_B = np.zeros(self.train_y.shape)
        self.splited_data = {
            0: (train_x_A, train_y_A),
            1: (train_x_B, train_y_B)
        }
        test_x_A = copy.deepcopy(self.test_x[:, :args["split"]])
        test_x_B = copy.deepcopy(self.test_x[:, args["split"]:])
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
            dataset = Dataset(item[0], item[1])
            self.data_loader[uid] = DataLoader(dataset, batch_size, shuffle=False)  # keep False
            # update num_steps/per round
            self.conf.fed_vertical["num_steps"] = len(self.data_loader[uid])

        # build test loader
        self.test_loader = {}
        for uid, item in self.splited_test_data.items():
            print("-> send testing data to client:{}, size:{}".format(uid, len(item[1])))
            dataset = Dataset(item[0], item[1])
            self.test_loader[uid] = DataLoader(dataset, batch_size, shuffle=False)

    def load_data(self, alpha=0.8):
        digits = datasets.load_digits()
        size = len(digits.target)
        per_label = int(size * (1 - alpha) / 10)

        # split data into train & test
        self.test_x, self.test_y = None, None
        self.train_x, self.train_y = None, None
        for label in range(10):
            idx = np.where(digits.target == label)[0]
            test_x = copy.deepcopy(digits.images[idx[:per_label]]) / 128.0 - 1
            test_y = copy.deepcopy(digits.target[idx[:per_label]])
            train_x = copy.deepcopy(digits.images[idx[per_label:]]) / 128.0 - 1
            train_y = copy.deepcopy(digits.target[idx[per_label:]])
            if self.test_x is not None:
                self.test_x = np.concatenate((self.test_x, test_x), axis=0)
                self.test_y = np.concatenate((self.test_y, test_y), axis=0)
                self.train_x = np.concatenate((self.train_x, train_x), axis=0)
                self.train_y = np.concatenate((self.train_y, train_y), axis=0)
            else:
                self.test_x = test_x
                self.test_y = test_y
                self.train_x = train_x
                self.train_y = train_y

        # random shuffle dataset
        size = len(self.train_x)
        idx = np.random.choice(size, size, replace=False)
        self.train_x = self.train_x[idx]
        self.train_y = self.train_y[idx]

        size = len(self.test_x)
        idx = np.random.choice(size, size, replace=False)
        self.test_x = self.test_x[idx]
        self.test_y = self.test_y[idx]

        # horizontal/vertical
        if self.conf.fed_partition == "horizontal":
            self.split_horizontal_data(self.conf.num_clients, self.conf.num_classes, self.conf.batch_size)
        elif self.conf.fed_partition == "vertical":
            print("-> vertically split data")
            self.split_vertical_data(self.conf.fed_vertical, self.conf.batch_size)



























