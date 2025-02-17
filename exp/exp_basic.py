
import logging
import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args,setting):
        self.args = args
        self.setting = setting
        self.logger = logging.getLogger()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
