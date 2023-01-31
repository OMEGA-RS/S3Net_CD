import numpy as np
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, data_path, label_path, indices_path=None, transform=None):
        self.images = np.load(data_path)
        self.ref = np.load(label_path)
        self.transform = transform

        if indices_path is not None:
            self.indices = np.load(indices_path)
            self.data = self.images[self.indices]
            self.label = self.ref[self.indices]
        else:
            self.data = self.images
            self.label = self.ref

        self.data1, self.data2 = np.split(self.data, 2, axis = -1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = self.label[index]

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2, y

    def __len__(self):
        return len(self.data)

