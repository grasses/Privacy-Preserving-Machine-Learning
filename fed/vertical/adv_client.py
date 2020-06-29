#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/29, homeway'

import copy
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from fed.vertical.client import Client


class AdversaryClient(Client):
    def __init__(self, uid, conf, data_loader, party, debug=False):
        super(AdversaryClient, self).__init__(uid, conf, data_loader, party, debug)

    def stop_round(self):
        self.last_x = copy.deepcopy(self.x)
        self.last_y = copy.deepcopy(self.y)

    def attack_gradient_leakage(self, last_grad, num_attack=300):
        """
        Implement of paper "Deep Leakage from Gradients"
        We use information in party A to inference party B's private training data.
        :return: inference information (i.e., local training image)
        """
        history = []
        w0 = self.global_params[0]["w"].detach().clone()
        w1 = self.global_params[1]["w"].detach().clone()
        x0 = self.x.detach().clone()
        y = self.y.detach().clone()
        x1 = torch.randn([self.x.shape[0], self.conf.fed_vertical["party"][1].num_features]).to(
            self.conf.device).requires_grad_(True)
        optimizer = torch.optim.Adam([x1], lr=0.1)

        for iters in range(num_attack):
            def closure():
                optimizer.zero_grad()
                # three steps for grad
                n = 1.0 / x0.shape[0]
                t = 0.25 * (x0 @ w0)
                u_prime = t - 0.5 * y

                v = 0.25 * (x1 @ w1)
                w = u_prime + v
                z = n * (w.t() @ x1).t()
                z_prime = n * (w.t() @ x0).t()

                dummy_grad = torch.cat([z_prime, z], axis=0)
                target_grad = torch.cat(last_grad, axis=0)
                loss = ((dummy_grad - target_grad) ** 2).sum()
                loss.backward(create_graph=True)
                return loss

            optimizer.step(closure)
            img_index = 0
            if iters % 10 == 0:
                current_loss = closure()
                print("-> iters={:d} loss={:.4f}".format(iters, current_loss.item()))
                a_part = (x0[img_index] + 1.0) * 128.0
                b_part = (x1[img_index] + 1.0) * 128.0
                img = torch.reshape(torch.cat((a_part, b_part), axis=0).cpu(), [8, 8])
                history.append(transforms.ToPILImage()(img))

        plt.figure(figsize=(18, 12))
        num_img = int(num_attack / 10)
        for i in range(num_img):
            plt.subplot(int(num_img / 10), 10, i + 1)
            plt.imshow(history[i])
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')
            plt.savefig(f"{self.conf.output_path}/{i}.jpg")











