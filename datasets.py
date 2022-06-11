from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchvision

test_folds = {
    0: [15, 16, 17, 18, 19],
    1: [10, 11, 12, 13, 14],
    2: [5, 6, 7, 8, 9],
    3: [0, 1, 2, 3, 4]
}

organs_code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22}

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(127.5, 127.5), # mapping des niveaux de gris dans [-1, 1]
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation(5),
    # torchvision.transforms.RandomCrop(17)
])

test_transform = torchvision.transforms.Normalize(127.5, 127.5)

class TrainDataset(Dataset):
    def __init__(self, fold):
        files = os.listdir(directory)
        files = [files[i] for i in range(20) if i not in test_folds[fold]]
        for file in files:
            file = os.fsdecode(file)
            if file.endswith(".csv"):
                labels = np.loadtxt(path + "\\" + file, delimiter=",", usecols=0)
                labels = np.vectorize(organs_code.get)(labels)
                labels_tensor = torch.from_numpy(labels)
                self.labels_tensor= labels_tensor.type(torch.FloatTensor)

                images_1D = np.loadtxt(path + "\\" + file, delimiter=",", usecols=np.arange(1,1090))
                images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
                images_tensor = torch.from_numpy(images_2D)
                self.images_tensor = images_tensor.type(torch.FloatTensor)

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        #self.sample = {'image': self.images_tensor[idx], 'label': self.labels_tensor[idx]}
        label = self.labels_tensor[idx]
        image = self.images_tensor[idx]
        image = train_transform(image)
        return (image, label)

class TestDataset(Dataset):
    def __init__(self, fold):
        files = os.listdir(directory)
        files = [files[i] for i in range(20) if i in test_folds[fold]]
        for file in files:
            file = os.fsdecode(file)
            if file.endswith(".csv"):
                del_idx = []
                labels = np.loadtxt(path + "\\" + file, delimiter=",", usecols=0)
                labels = np.vectorize(organs_code.get)(labels)
                for idx, label in enumerate(labels):
                    if int(label) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]:
                        del_idx.append(idx)
                labels = np.delete(labels, del_idx)
                labels_tensor = torch.from_numpy(labels)
                self.labels_tensor= labels_tensor.type(torch.FloatTensor)

                images_1D = np.loadtxt(path + "\\" + file, delimiter=",", usecols=np.arange(1,1090))
                images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
                images_tensor = torch.from_numpy(images_2D)
                self.images_tensor = images_tensor.type(torch.FloatTensor)

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        #self.sample = {'image': self.images_tensor[idx], 'label': self.labels_tensor[idx]}
        label = self.labels_tensor[idx]
        image = self.images_tensor[idx]
        image = test_transform(image)
        return (image, label)