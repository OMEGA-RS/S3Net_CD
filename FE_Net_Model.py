import os
import torch
import torch.nn as nn
from Net_Model import SiameseNet

MODEL_PATH = os.path.join('Data\Guangzhou', 'Pretrained_model.pt')

class FE_Net(nn.Module):

    def __init__(self):
        super(FE_Net, self).__init__()

        Pretrained_Model = SiameseNet()
        Pretrained_Model.load_state_dict(torch.load(MODEL_PATH))

        self.features = Pretrained_Model

    def forward(self, x):

        x1, x2, x3, x4 = self.features.input_forward(x)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        x3 = x3.view(x3.size()[0], -1)
        x4 = x4.view(x4.size()[0], -1)

        return x1, x2, x3, x4

