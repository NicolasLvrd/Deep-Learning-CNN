import torch
import os
import torch.nn.functional as F
import numpy as np
import math

import matplotlib.pyplot as plt
from torch import TracingState, nn, tensor, tensor_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

docCount = 1;

#> hyperparamètres
learning_rate = 1e-3
batch_size = 64
epochs = 1


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#> chargement du de données
datasets = []

code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22}

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

'''
cpt = 0
all_images_list = [ [] for i in range(23)]
all_labels_list = [ [] for i in range(23)]
datasets_by_label = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print(filename)
        for i in range(0, 10000, 1000):
            labels = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=0, skiprows=i, max_rows=1000)
            labels = np.vectorize(code.get)(labels)
            all_labels_list[labels[0]].append(labels)
 
            Array1d_images = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=np.arange(1,1090), skiprows=i, max_rows=1000)
            Array1d_images = np.floor(Array1d_images)
            #print(Array1d_images)
            Array2d_images = Array1d_images.reshape(Array1d_images.shape[0], 1, 33, 33)
            all_images_list[labels[0]].append(Array2d_images)

        cpt += 1
        if(cpt == docCount):
            break
print("YUP !")

unique_labels = np.array([])
for i in range( len(all_labels_list) ):
    labels_set = np.unique(all_labels_list[i])
    unique_labels = np.concatenate((unique_labels, labels_set))

labels_set = np.unique(unique_labels)

dic = {}
for i in range(len(labels_set)):
    dic[labels_set[i]] = i

for i in range(23):
    try:
        array_labels = np.concatenate(all_labels_list[i])
        array_labels = np.vectorize(dic.get)(array_labels)
        #print(array_labels)
        tensor_labels = torch.from_numpy(array_labels)
        tensor_labels = tensor_labels.type(torch.FloatTensor)
        print(tensor_labels)

        array_images = np.concatenate(all_images_list[i])
        tensor_images = torch.from_numpy(array_images)
        tensor_images = tensor_images.type(torch.FloatTensor)
        #print(tensor_images.shape)

        datasets_by_label.append(TensorDataset(tensor_images,tensor_labels))
    except:
        continue

print("All datasets have been created.")
'''
#DATASETS CREATION BIS
all_labels_list = []
all_images_list = []

cpt = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print("Reading ", filename)
        labels = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=0)
        all_labels_list.append(labels)

        Array1d_images = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=np.arange(1,1090))
        #Array1d_images = np.floor(Array1d_images)
        Array2d_images = Array1d_images.reshape(Array1d_images.shape[0], 1, 33, 33)
        all_images_list.append(Array2d_images)

        cpt += 1
        if(cpt == docCount):
            break

all_labels = np.concatenate(all_labels_list)
all_images = np.concatenate(all_images_list)

unique_labels = np.unique(all_labels)

dic = {}
for i in range(len(unique_labels)):
    dic[unique_labels[i]] = i

all_labels = np.vectorize(dic.get)(all_labels)

tensor_labels = torch.from_numpy(all_labels)   
tensor_labels = tensor_labels.type(torch.FloatTensor)
#tensor_labels = tensor_labels.to(device)

tensor_images = torch.from_numpy(all_images)
tensor_images = tensor_images.type(torch.FloatTensor)
#tensor_images = tensor_images.to(device)

dataset = TensorDataset(tensor_images,tensor_labels) 
print("Done.")

#DATASETS CREATION BIS (END)

train_sampler, valid_sampler = torch.utils.data.random_split((dataset), [math.floor(len(dataset)*0.8), len(dataset)-math.floor(len(dataset)*0.8)])

#print("=======>", train_sampler.data.shape)

#> initialisation des dataloaders
train_dataloader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)


dataiter = iter(train_dataloader)
images, labels = dataiter.next()
print("IMAGES :", images.shape)
print("LABELS :", labels.shape)
img = images[0][0]
img = img.cpu()
npimg = img.numpy()
plt.imshow(npimg)
plt.show()
print(labels[0])

#for i, (input, target) in enumerate(test_dataloader):
#    print(target)*


#> définition du réseau
class Net(nn.Module):
    '''
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(33*33, 256),
            nn.ReLU(),
            nn.Conv2d(256, 6, 5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 23),
        )
    '''
    '''
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(256, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 16, 5),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    '''

    def __init__(self):
        super().__init__()
        # 1 x 33 x 33
        self.conv1 = nn.Conv2d(1, 20, 5) #, stride=1, padding=0, dilation=1
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20, 40, 3) #, stride=1, padding=0, dilation=1
        self.conv3 = nn.Conv2d(40, 60, 2) #, stride=1, padding=0, dilation=1
        self.fc1 = nn.Linear(240, 180)
        self.fc2 = nn.Linear(180, 75)
        self.fc3 = nn.Linear(75, 23)

    def forward(self, x):
        print("start :", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("after conv1 :", x.shape)
        '''
        x=self.conv1(x)
        print("after conv1 :", x.shape)
        x = self.pool(F.relu(x))
        print("after pool :", x.shape)
        '''
        x = self.pool(F.relu(self.conv2(x)))
        
        print("after conv2 :", x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print("after conv3 :", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("after flatten :", x.shape)
        x = F.relu(self.fc1(x))
        #print("after fc1 :", x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print("end")
        return x

model = Net()
model = Net().to(device)
model.to(torch.float)

#> boucle d'apprentissage
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct, train_loss = 0, 0
    dataloader.data.to(device)
    for batch, (X, y) in enumerate(dataloader):
        #print(X.shape)
        # Compute prediction and loss
        pred = model(X)
        print("----------------------------------------")
        print(pred.shape)
        print(y.shape)
        print(pred)
        print(y)
        loss = loss_fn(pred, y.long())
        train_loss += loss_fn(pred, y.long()).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        '''
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Accuracy: {(100*correct):>0.1f}%")
        '''
    train_loss /= len(dataloader)
    correct /= size
    print("TRAIN LOOP")
    print("    loss: ", train_loss)
    print("    accuracy: ", 100*correct)
    dataloader.data.to('cpu')

#> boucle de test
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print("TEST LOOP")
    print("    loss: ", test_loss)
    print("    accuracy: ", 100*correct)
    print("")

#> fonction de perte et algorythme d'optimisation
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#> execution de l'apprentissage et des tests
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")