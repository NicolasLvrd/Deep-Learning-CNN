import torch
from datasets import *

for fold in range(4):
    print("Fold: ", fold)

    train_dataset = TrainDataset(fold)
    torch.save(train_dataset, "./data/train/train_fold"+str(fold)+".pt")

    test_dataset = TestDataset(fold)
    torch.save(test_dataset, "./data/test/test_fold"+str(fold)+".pt")