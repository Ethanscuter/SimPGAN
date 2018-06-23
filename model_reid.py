import torch.nn as nn
import torch
from pretrain import Identify_net
import torch.backends.cudnn as cudnn
from pretrain_siamese import Siamese
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

source_model_path = './market_softmax_pretrain.pkl'


class DReid(nn.Module):
    def __init__(self, class_count):
        super().__init__()

        self.model = Identify_net(class_count)
        self.model = torch.nn.DataParallel(self.model).cuda()
        cudnn.benchmark = True
        self.model.load_state_dict(torch.load(source_model_path))

    def forward(self, x):
        x, out = self.model(x)
        return x, out
        

source_siamese_model_path = './market_pair_pretrain.pkl'


class DSiamese(nn.Module):
    def __init__(self, class_count):
        super().__init__()

        base_model = Identify_net(class_count)
        base_model = torch.nn.DataParallel(base_model).cuda()
        cudnn.benchmark = True
        base_model.load_state_dict(torch.load(source_model_path))

        self.model = Siamese(base_model)
        self.model = torch.nn.DataParallel(self.model).cuda()
        cudnn.benchmark = True

        self.model.load_state_dict(torch.load(source_siamese_model_path))

    def forward(self, input1, input2):
        feature, out1, out2, out = self.model(input1, input2)
        return feature, out1, out2, out
