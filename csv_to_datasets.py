import numpy as np
import torch
import os
import torchvision
from torch.utils.data import TensorDataset


docCount = 20 # nombre de patients considérés

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(127.5, 127.5), # mapping des niveaux de gris dans [-1, 1]
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20)
])

test_transform = torchvision.transforms.Normalize(127.5, 127.5)

# lecture des données et création des datasets
code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22} # dictionnaire des labels originaux avec des labels dans [0;22] pour utiliser CE loss

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

patients = [] # datasets de chaque patient

cpt = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print("Reading ", filename)

        labels = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=0)
        labels = np.vectorize(code.get)(labels) # mapping des labels
        wished_idx = []
        for idx, label in enumerate(labels):
            if int(label) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]:
                wished_idx.append(idx)
        labels_test = np.delete(labels, wished_idx)
        labels_tensor = torch.from_numpy(labels)
        labels_tensor_test = torch.from_numpy(labels_test)
        labels_tensor = labels_tensor.type(torch.FloatTensor)
        labels_tensor_test = labels_tensor_test.type(torch.FloatTensor)
        # labels_tensor = labels_tensor.to(device)

        images_1D = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=np.arange(1,1090))
        images_1D_test = np.delete(images_1D, wished_idx, axis=0)
        images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
        images_2D_test = images_1D_test.reshape(images_1D_test.shape[0], 1, 33, 33)
        images_tensor = torch.from_numpy(images_2D)
        images_tensor_test = torch.from_numpy(images_2D_test)
        images_tensor = images_tensor.type(torch.FloatTensor)
        images_tensor_test = images_tensor_test.type(torch.FloatTensor)
        images_tensor = train_transform(images_tensor)
        images_tensor_test = test_transform(images_tensor_test)
        # images_tensor = images_tensor.to(device)

        torch.save(TensorDataset(images_tensor, labels_tensor), "./data/train_patient"+str(cpt))
        # patients.append(TensorDataset(images_tensor, labels_tensor))
        torch.save(TensorDataset(images_tensor_test, labels_tensor_test), "./data/test_patient"+str(cpt))

        cpt += 1
        if(cpt == docCount):
            break