import numpy as np
import torch
import os
import torchvision


docCount = 20 # nombre de patients considérés

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(127.5, 127.5), # mapping des niveaux de gris dans [-1, 1]
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(5),
    #torchvision.transforms.RandomResizedCrop(33)
])

test_transform = torchvision.transforms.Normalize(127.5, 127.5)

# lecture des données et création des datasets
code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22} # dictionnaire des labels originaux avec des labels dans [0;22] pour utiliser CE loss

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

train_labels_list = [[] for i in range(4)]
test_labels_list = [[] for i in range(4)]
train_images_list = [[] for i in range(4)]
test_images_list = [[] for i in range(4)]

cpt = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print("Reading ", filename)
        del_idx = []
        labels = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=0)
        labels = np.vectorize(code.get)(labels) # mapping des labels
        for idx, label in enumerate(labels):
            if int(label) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]:
                del_idx.append(idx)
        labels_test = np.delete(labels, del_idx)
        labels_tensor, labels_tensor_test = torch.from_numpy(labels), torch.from_numpy(labels_test)
        labels_tensor, labels_tensor_test = labels_tensor.type(torch.FloatTensor), labels_tensor_test.type(torch.FloatTensor)
        train_labels_list[cpt // 5].append(labels_tensor), test_labels_list[cpt // 5].append(labels_tensor_test)

        images_1D = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=np.arange(1,1090))
        images_1D_test = np.delete(images_1D, del_idx, axis=0)
        images_2D, images_2D_test = images_1D.reshape(images_1D.shape[0], 1, 33, 33), images_1D_test.reshape(images_1D_test.shape[0], 1, 33, 33)
        images_tensor, images_tensor_test = torch.from_numpy(images_2D), torch.from_numpy(images_2D_test)
        images_tensor, images_tensor_test = images_tensor.type(torch.FloatTensor), images_tensor_test.type(torch.FloatTensor)
        images_tensor, images_tensor_test = train_transform(images_tensor), test_transform(images_tensor_test)
        train_images_list[cpt // 5].append(images_tensor), test_images_list[cpt // 5].append(images_tensor_test)
        
        '''
        torch.save(TensorDataset(images_tensor, labels_tensor), "./data/train_patient"+str(cpt))
        # patients.append(TensorDataset(images_tensor, labels_tensor))
        torch.save(TensorDataset(images_tensor_test, labels_tensor_test), "./data/test_patient"+str(cpt))
        '''

        cpt += 1
        if(cpt == docCount):
            break

for i in range(4):
    train_labels_cat = torch.cat(train_labels_list[i], dim=0)
    print("train_labels_cat.shape", train_labels_cat.shape)
    train_images_cat = torch.cat(train_images_list[i], dim=0)
    print("train_images_cat.shape", train_images_cat.shape)
    test_labels_cat = torch.cat(test_labels_list[i], dim=0)
    print("test_labels_cat.shape", test_labels_cat.shape)
    test_images_cat = torch.cat(test_images_list[i], dim=0)
    print("test_images_cat.shape", test_images_cat.shape)

    torch.save(train_labels_cat, "./data/train/labels/group"+str(i))
    torch.save(train_images_cat, "./data/train/images/group"+str(i))
    torch.save(test_labels_cat, "./data/test/labels/group"+str(i))
    torch.save(test_images_cat, "./data/test/images/group"+str(i))
