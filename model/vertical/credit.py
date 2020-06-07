
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from model import Model


class PartyModel(Model):
    def __init__(self, num_features, num_output, name="", created_time="", debug=False):
        super(PartyModel, self).__init__(name, created_time)
        self.w = self.xavier_init([num_features, num_output])
        self.debug = debug

    def step1(self, x, y):
        t = 0.25 * x @ self.w

        #print("-> step1, w={}".format(self.w.shape))
        #print("-> t.shape={}\n".format(t.shape))

        u_prime = t - 0.5 * y
        return u_prime

    def step2(self, x, u_prime):
        #print("-> step2, w={}".format(self.w.shape))
        v = 0.25 * x @ self.w
        #print("-> step2, v={}".format(v.shape))

        w = u_prime + v
        #print("-> step2, x={}, w={}\n".format(x.shape, w.shape))
        z = w.t() @ x
        return w.t(), z

    def step3(self, x, w):
        #print("-> step3, x={}, w={}".format(x.shape, w.shape))
        z_prime = w @ x
        #print("-> step3, z_prime={}".format(z_prime.shape))
        return z_prime

    def forward(self, x):
        x = x @ self.w
        x = torch.sigmoid(x)
        return x

