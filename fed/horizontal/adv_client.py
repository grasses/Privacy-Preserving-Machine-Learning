#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/5, homeway'

import os
import copy
import torch.optim as optim
import torch.nn.functional as F
from fed.horizontal.client import Client
from model.horizontal import Model
from model.horizontal.generator import Generator

import torch
import numpy as np
from torch.autograd import Variable

class AdversaryClient(Client):
    def __init__(self, uid, conf, data_loader):
        super(AdversaryClient, self).__init__(uid, conf, data_loader)
        self.D = Model(self.conf.num_features, self.conf.num_classes).to(self.conf.device)
        self.G = Generator(num_input=10, num_output=self.conf.num_features).to(self.conf.device)

    def train_GAN(self, parameters, step, lr_G=0.001):
        # copy model to D
        self.D.copy_params(parameters)
        optimizer_G = optim.Adam(self.G.parameters(), lr=lr_G)

        G_dataset = None
        self.G.train()
        for i in range(step):
            z = Variable(torch.randn(self.conf.batch_size, 9)).to(self.conf.device)
            c = Variable(torch.ones([self.conf.batch_size, 1])).to(self.conf.device)

            G_sample = self.G(z, c)
            D_fake = self.D(G_sample.detach())

            # only optimized G
            optimizer_G.zero_grad()
            loss = -torch.mean(torch.exp(1 - D_fake)) # KL (Kullback–Leibler) divergence
            loss.backward()
            optimizer_G.step()

            if i == 0:
                G_dataset = G_sample.detach().cpu().numpy()
            else:
                G_dataset = np.concatenate((G_dataset, G_sample.detach().cpu().numpy()), axis=0)

            if i > 0 and i % 10 == 0:
                print("-> Generator, step:{} loss:{}".format(i, loss))

            np.save(os.path.join(self.conf.output_path, "G_sample.npy"), G_dataset)
        return G_dataset

    def update(self, parameters):
        print("\n<-------------------- Adversary client at attack mode -------------------->")

        self.global_params = copy.deepcopy(parameters)
        G_dataset = self.train_GAN(parameters=self.global_params, step=1)

        self.model.copy_params(self.global_params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.conf.learning_rate)
        self.model.train()
        for e in range(self.conf.fed_epoch):
            train_loss, train_acc = 0, 0
            for step, (x, y) in enumerate(self.data_loader):
                x, y = x.to(self.conf.device), y.to(self.conf.device)
                self.optimizer.zero_grad()
                logists = self.model(x)
                loss = F.cross_entropy(logists, y)
                loss.backward()

                pred = logists.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(y.view_as(pred)).sum().item()
                train_loss += F.cross_entropy(logists, y, reduction='sum').item()
                self.optimizer.step()
        print("-> client{:d} finish! train_loss={:.3f} train_acc={:.1f}%".format(
            self.uid,
            train_loss / len(self.data_loader.dataset),
            100 * train_acc / len(self.data_loader.dataset))
        )
        print("<-------------------- Adversary attack ending -------------------->\n")

        if self.conf.fed_horizontal["encrypt_weight"]:
            return self.encrypted_model(self.model)
        return self.model.state_dict()