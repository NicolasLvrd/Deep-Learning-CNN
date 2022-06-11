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

class TrainDataset(Dataset):
    def __init__(self, fold):
        print(f"Fold {fold} : TRAIN SET")
        labels_list = [train_labels_list[i] for i in range(20) if i not in test_folds[fold]]
        images_list = [train_images_list[i] for i in range(20) if i not in test_folds[fold]]

        self.labels_tensor = torch.cat(labels_list)
        self.images_tensor = torch.cat(images_list)

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
        print(f"Fold {fold} : TEST SET")

        labels_list = [test_labels_list[i] for i in range(20) if i in test_folds[fold]]
        images_list = [test_images_list[i] for i in range(20) if i in test_folds[fold]]

        self.labels_tensor = torch.cat(labels_list)
        self.images_tensor = torch.cat(images_list)

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        #self.sample = {'image': self.images_tensor[idx], 'label': self.labels_tensor[idx]}
        label = self.labels_tensor[idx]
        image = self.images_tensor[idx]
        image = test_transform(image)
        return (image, label)

organs_code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22}

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(127.5, 127.5), # mapping des niveaux de gris dans [-1, 1]
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation((0, 5)),
    # torchvision.transforms.RandomCrop(17)
    # torchvision.transforms.RandomResizedCrop(33)
])

test_transform = torchvision.transforms.Normalize(127.5, 127.5)

if __name__ == '__main__':
    train_labels_list = []
    train_images_list = []
    test_labels_list = []
    test_images_list = []

    files = os.listdir(directory)
    for file in files:
        file = os.fsdecode(file)
        if file.endswith(".csv"):
            print(f"Reading {file}")
            # ------- LABELS ------- #
            labels = np.loadtxt(path + "\\" + file, delimiter=",", usecols=0)
            labels = np.vectorize(organs_code.get)(labels)

            # train #
            train_labels_tensor = torch.from_numpy(labels)
            train_labels_tensor= train_labels_tensor.type(torch.FloatTensor)

            # test #
            del_idx = []
            for idx, label in enumerate(labels):
                if int(label) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]:
                        del_idx.append(idx)
            labels = np.delete(labels, del_idx)
            test_labels_tensor = torch.from_numpy(labels)
            test_labels_tensor = test_labels_tensor.type(torch.FloatTensor)

            # ------- IMAGES ------- #
            images_1D = np.loadtxt(path + "\\" + file, delimiter=",", usecols=np.arange(1,1090))

            # train #
            images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
            train_images_tensor = torch.from_numpy(images_2D)
            train_images_tensor = train_images_tensor.type(torch.FloatTensor)

            # test #
            images_1D = np.delete(images_1D, del_idx, axis=0)
            images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
            test_images_tensor = torch.from_numpy(images_2D)
            test_images_tensor = test_images_tensor.type(torch.FloatTensor)

            # ------- SAVING TENSORS ------- #
            train_labels_list.append(train_labels_tensor)
            train_images_list.append(train_images_tensor)
            test_labels_list.append(test_labels_tensor)
            test_images_list.append(test_images_tensor)

    for fold in range(4):
        train_dataset = TrainDataset(fold)
        torch.save(train_dataset, "./data/train/train_fold"+str(fold)+".pt")

        test_dataset = TestDataset(fold)
        torch.save(test_dataset, "./data/test/test_fold"+str(fold)+".pt")