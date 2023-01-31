import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNet(nn.Module):

    def __init__(self, patch_size=22, channel_num=3):
        super(SiameseNet, self).__init__()

        self.patch_size = patch_size
        self.channel_num = channel_num

        self.features = models.resnet18(pretrained=True)

        self.features.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)

        for name, value in self.features.named_parameters():
            if (name != 'conv1.weight') and (name != 'conv1.bias'):
                value.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(in_features=2304, out_features=512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=3)

        )

    def input_forward(self, x):

        x1 = self.features.conv1(x)

        x2 = self.features.bn1(x1)
        x2 = self.features.relu(x2)

        x3 = self.features.maxpool(x2)

        x4 = self.features.layer1(x3)

        return x1, x2, x3, x4

    def forward(self, x1, x2):
        batch_size = x1.size()[0]

        # 特征相减
        x1 = self.input_forward(x1)[3]
        x2 = self.input_forward(x2)[3]
        x = abs(torch.sub(x1, x2))
        x = x.flatten().view(batch_size, -1)

        x = self.fc(x)

        return x

