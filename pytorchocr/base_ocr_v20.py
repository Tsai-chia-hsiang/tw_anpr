import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch

from .modeling.architectures.base_model import BaseModel

class BaseOCRV20(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        if self.config['Head']['name'] == 'MultiHead' :
            self.config['Head']['out_channels_list'] = {
            'CTCLabelDecode': kwargs['char_num'],
            'SARLabelDecode': kwargs['char_num'] + 2,
            'NRTRLabelDecode': kwargs['char_num'] + 3
        }

        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)


    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        return weights

    def get_out_channels(self, weights):
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))


    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))


    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k,v in self.net.state_dict().items():
            print('{}----{}'.format(k,type(v)))


    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer

