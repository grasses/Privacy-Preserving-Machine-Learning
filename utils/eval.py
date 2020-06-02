#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import torch
import torch.nn.functional as F

class Evaluation():
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.test_loader = data.test_loader
        self.test_size = len(self.test_loader.dataset)

    def eval_test(self, model):
        test_loss, test_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for (x, y) in self.test_loader:
                x, y = x.to(self.conf.device), y.to(self.conf.device)
                logists = model(x)
                test_loss += F.cross_entropy(logists, y, reduction='sum').item()
                pred = logists.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(y.view_as(pred)).sum().item()
            test_loss /= self.test_size
            accuracy = 100. * test_correct / self.test_size
            print("-> Test set: Average loss: {:.4f}, Accuracy: {:d}/{:d} ({:.3f}%)".format(
                test_loss, test_correct,
                self.test_size,
                accuracy)
            )
        return float(accuracy)