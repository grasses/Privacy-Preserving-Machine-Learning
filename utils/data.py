#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import copy
import numpy as np

def split_data(data, label, num_clients, num_classes, alpha=0.9):
    np.random.seed(100)
    data_splited = {}
    class_size = len(data) / num_classes
    sample_prob = np.random.dirichlet(np.array(num_clients * [alpha]))

    for class_ in range(num_classes):
        data_by_class = copy.deepcopy(
            data[np.argwhere(label == class_).squeeze()])
        np.random.shuffle(data_by_class)
        j = 0
        for client in range(num_clients):
            i = min(len(data_by_class), j)
            num_samples = int(round(class_size * sample_prob[client]))
            j += num_samples
            sample_image = data_by_class[i:j]
            sample_label = np.array([class_] * len(sample_image))
            if class_ == 0:
                data_splited[client] = {}
                data_splited[client][0] = sample_image
                data_splited[client][1] = sample_label
            else:
                data_splited[client][0] = np.concatenate((data_splited[client][0], sample_image), axis=0)
                data_splited[client][1] = np.concatenate((data_splited[client][1], sample_label), axis=0)
    return data_splited


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            img, target = t(x, y)
        return img, target