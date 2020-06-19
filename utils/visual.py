#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/15, homeway'

import os
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
from config.conf import Conf
plt.ioff()

class Visual():
    def __init__(self, conf=Conf):
        self.conf = conf

    def plot_gan_distributions(self, x, y, name="GAN_estimate"):
        red = y == 0
        green = y == 1

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_transformed = tsne.fit_transform(x)

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)

        ax.scatter(x_transformed[red, 0], x_transformed[red, 1], c="r")
        ax.scatter(x_transformed[green, 0], x_transformed[green, 1], c="g")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        fpath = os.path.join(self.conf.output_path, "{:s}.pdf".format(name))
        print("-> save fig in: {}".format(fpath))
        plt.savefig(fpath, format='pdf')
