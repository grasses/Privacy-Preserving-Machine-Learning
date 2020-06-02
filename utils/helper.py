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

    def weights_avg(self, model, collection):
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
                res[key] += (1.0 * collection[id][key])
            res[key] = 1.0 * (res[key].float_precision()) / size
        return collections.OrderedDict(res)