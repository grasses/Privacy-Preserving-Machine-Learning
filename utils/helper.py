# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import os
import copy
import collections
import argparse
from config.conf import Conf

class Helper():
    def __init__(self, conf=None):
        self.conf = conf
        self.args = self.get_args()
        if not conf:
            self.conf = self.init()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="conf")
        return parser.parse_args()

    def init(self):
        self.conf = __import__("config." + self.args.config, globals(), locals(), ["Conf"]).Conf
        # build setup path
        init_path = [self.conf.model_path, self.conf.output_path]
        for path in init_path:
            if not os.path.exists(path):
                os.makedirs(path)
        return self.conf

    def weights_avg(self, model, collection, fix_precision=False):
        '''
        :param model: dict() {key: weight,...}
        :param collection: dict() {id: {key: weight},...}
        :return: weights aggregation
        '''
        res = {}
        size = len(collection)
        for (key, value) in model.items():
            res[key] = value.clone()
            for id in collection.keys():
                if fix_precision:
                    value = copy.deepcopy(collection[id][key].float_precision())
                else:
                    value = copy.deepcopy(collection[id][key])
                res[key] += (1.0 * value)
            res[key] = 1.0 * (res[key]) / size
        return collections.OrderedDict(res)