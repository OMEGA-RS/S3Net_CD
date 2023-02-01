import os
import copy
import torch
import torch.nn as nn
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
from scipy.io import savemat
from Data_Loading import MyDataSet
from Net_Model import SiameseNet

warnings.filterwarnings("ignore")
torch.manual_seed(1)
torch.cuda.manual_seed(1)

PATCH_SIZE = 22
CHANNEL_NUM = 3
BATCH_SIZE = 64
MAX_POCH = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

DATA_DIR = 'Data\Guangzhou'

DATA_PATH = os.path.join(DATA_DIR, 'pretext_data.npy')
LABEL_PATH = os.path.join(DATA_DIR, 'pretext_label.npy')

TRAIN_INDICES = os.path.join(DATA_DIR, 'train_indices.npy')
VAL_INDICES = os.path.join(DATA_DIR, 'val_indices.npy')

SAVE_MODEL_PATH = os.path.join(DATA_DIR, 'Pretrained_model.pt')


train_dataset = MyDataSet(DATA_PATH, LABEL_PATH, TRAIN_INDICES,
                          transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize([0.301, 0.404, 0.284], [0.278, 0.204, 0.207])
                          ]))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = MyDataSet(DATA_PATH, LABEL_PATH, VAL_INDICES,
                        transform=transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.301, 0.404, 0.284], [0.278, 0.204, 0.207])
                        ]))

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_dataset_label = train_dataset.label
val_dataset_label = val_dataset.label

train_set_num = len(train_dataset_label)
val_set_num = len(val_dataset_label)

net = SiameseNet()
net.to(device)

p0 = (train_dataset_label == 0).sum() / train_set_num
p1 = (train_dataset_label == 1).sum() / train_set_num
p2 = (train_dataset_label == 2).sum() / train_set_num

class_weight = torch.FloatTensor([p0, p1, p2]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weight)

net_params = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.Adam(net_params, lr=1e-3, betas=(0.5, 0.9), weight_decay=3e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5)

training_loss = []
validation_loss = []
KC = 0.0

for epoch in range(1, MAX_POCH + 1):
    net.train()

    train_loss = 0.0
    train_pred_value = []
    train_real_value = []

    for idx,data in enumerate(train_dataloader):
        x1, x2, y = data
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        y = y.long()

        optimizer.zero_grad()
        output_result = net(x1, x2)

        loss = criterion(output_result, y)
        train_pred = torch.argmax(output_result, dim=1)

        loss.backward()
        optimizer.step()

        train_pred_value.append(train_pred)
        train_real_value.append(y)

        train_loss += loss.item()

    train_pred_value = torch.cat(train_pred_value).detach().cpu().numpy().squeeze()
    train_real_value = torch.cat(train_real_value).detach().cpu().numpy()

    train_KC = metrics.cohen_kappa_score(train_pred_value, train_real_value)
    print("-" * 30)
    print("Epoch {}/{} train set KC = : {:.6f}".format(epoch, MAX_POCH, train_KC))
    print("Epoch {}/{} train loss = : {:.6f}".format(epoch, MAX_POCH, train_loss / len(train_pred_value)))

    training_loss.append(train_loss / train_set_num)
    scheduler.step(train_loss)

    with torch.no_grad():
        net.eval()

        val_loss = 0.0
        predict_result = []
        real_result = []

        for idx, data in enumerate(val_dataloader):
            x1, x2, real_value = data
            x1 = x1.to(device)
            x2 = x2.to(device)

            real_value = real_value.to(device)
            real_value = real_value.long()

            net_output = net(x1, x2)
            loss = criterion(net_output, real_value)
            val_pred = torch.argmax(net_output, dim=1)

            predict_result.append(val_pred)
            real_result.append(real_value)

            val_loss += loss.item()

        validation_loss.append(val_loss / val_set_num)

        predict_result = torch.cat(predict_result).cpu().numpy().squeeze()
        real_result = torch.cat(real_result).cpu().numpy()

        val_KC = metrics.cohen_kappa_score(real_result, predict_result)
        print("\nEpoch {}/{} val set KC = : {:.6f}".format(epoch, MAX_POCH, val_KC))
        print("Epoch {}/{} val loss = : {:.6f}".format(epoch, MAX_POCH, val_loss / len(predict_result)))

    if val_KC > KC:
        KC = val_KC
        best_model_wts = copy.deepcopy(net.state_dict())

torch.save(best_model_wts, SAVE_MODEL_PATH)
