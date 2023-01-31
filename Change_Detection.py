import os
import torch
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
from skimage import io,filters
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import savemat
from sklearn import metrics
from Data_Loading import MyDataSet
from FE_Net_Model import FE_Net

warnings.filterwarnings("ignore")
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

BATCH_SIZE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

DATA_DIR = 'Data\Guangzhou'

DATA_PATH = os.path.join(DATA_DIR, 'image_data.npy')
REF_PATH = os.path.join(DATA_DIR, 'ref_label.npy')
SP_MASK_PATH = os.path.join(DATA_DIR, 'seg_label.npy')

image_dataset = MyDataSet(DATA_PATH, REF_PATH, transform=transforms.ToTensor())
image_dataset = MyDataSet(DATA_PATH, REF_PATH, transform=transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.301, 0.404, 0.284], [0.278, 0.204, 0.207])
                         ]))

image_dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False)


net = FE_Net()
net = net.to(device)


V1 = [[],[],[],[],[]]
V2 = [[],[],[],[],[]]

with torch.no_grad():
    net.eval()

    for idx, data in enumerate(image_dataloader):
        x1, x2, _y = data
        x1 = x1.to(device)
        x2 = x2.to(device)

        output1 = net(x1)
        output2 = net(x2)

        for k in range(4):
            tmp1 = output1[k].cpu().numpy()
            tmp1 = tmp1.flatten()
            V1[k].append(tmp1)

            tmp2 = output2[k].cpu().numpy()
            tmp2 = tmp2.flatten()
            V2[k].append(tmp2)

segment_img = np.load(SP_MASK_PATH)
segment_idx = np.unique(segment_img)

Height, Width = segment_img.shape
X_D = np.zeros((5, Height, Width))
X_D[0] = io.imread(os.path.join(DATA_DIR, 'DI.png'))

for k in range(4):
    cmp = np.zeros((Height, Width))

    F1 = normalization(np.array(V1[k]))
    F2 = normalization(np.array(V2[k]))
    F_D = np.sqrt(np.square(F2 - F1).sum(axis=1))

    for num in segment_idx:
        cmp[segment_img == num] = F_D[num]

    X_D[k + 1] = (normalization(cmp) * 255).astype(np.uint8)

Alpha = [0.4, 0.1, 0.3, 0.1, 0.1]

diff_map = np.zeros((Height, Width))
for k in range(5):
    diff_map += Alpha[k] * X_D[k]
diff_map = diff_map.astype(np.uint8)

threshold = filters.threshold_otsu(diff_map)
binary_map = ((diff_map >= threshold) * 255).astype(np.uint8)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

temp = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel1)
cm_map = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel2)
