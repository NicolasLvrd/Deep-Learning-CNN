import torch
import os
import torch.nn.functional as F
import numpy as np
import math
import ignite
#import matplotlib.pyplot as plt
from torch import TracingState, nn, tensor, tensor_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


docCount = 20;

#> hyperparamètres
learning_rate = 1e-3
batch_size = 64
epochs = 2


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#> chargement du de données
code = {1247:0 , 1302:1 , 1326:2 , 170:3 , 187:4 , 237:5 , 2473:6 , 29193:7 , 29662:8 , 29663:9 , 30324:10 , 30325:11 , 32248:12 , 32249:13 , 40357:14 , 40358:15, 480:16 , 58:17 , 7578:18, 86:19 , 0:20 , 1:21 , 2:22}

path = "data\CTce_ThAb_b33x33_n1000_8bit"
directory = os.fsencode(path)

patients = []
groups = []

cpt = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print("Reading ", filename)

        labels = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=0) #, max_rows=5000
        labels = np.vectorize(code.get)(labels)

        '''
        labels_set = np.unique(labels)

        dic = {}
        for i in range(len(labels_set)):
            dic[labels_set[i]] = i

        labels = np.vectorize(dic.get)(labels)
        print(labels)

        '''
        labels_tensor = torch.from_numpy(labels)
        labels_tensor = labels_tensor.type(torch.FloatTensor)
        labels_tensor = labels_tensor.to(device)

        images_1D = np.loadtxt(path + "\\" + filename, delimiter=",", usecols=np.arange(1,1090)) #, max_rows=5000
        images_2D = images_1D.reshape(images_1D.shape[0], 1, 33, 33)
        images_tensor = torch.from_numpy(images_2D)
        images_tensor = images_tensor.type(torch.FloatTensor)
        images_tensor = images_tensor.to(device)

        patients.append(TensorDataset(images_tensor, labels_tensor))

        cpt += 1
        if(cpt == docCount):
            break

#patients = torch.utils.data.ConcatDataset(patients) // à ne pas considérer
#train_sampler, valid_sampler = torch.utils.data.random_split((patients), [math.floor(len(patients)*0.8), len(patients)-math.floor(len(patients)*0.8)])

for i in range(4):
    groups.append( torch.utils.data.ConcatDataset((patients[5*i], patients[5*i+1], patients[5*i+2], patients[5*i+3], patients[5*i+4])) )


#> définition du réseau
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 33 x 33
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.conv3 = nn.Conv2d(40, 60, 2)
        self.fc1 = nn.Linear(240, 180)
        self.fc2 = nn.Linear(180, 75)
        self.fc3 = nn.Linear(75, 23)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#> boucle d'apprentissage
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.long())
        train_loss += loss_fn(pred, y.long()).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= len(dataloader)
    correct /= size
    print("TRAIN LOOP")
    print("    loss: ", train_loss)
    print("    accuracy: ", 100*correct)
    return np.array([train_loss, correct])

#> boucle de validation
def valid_loop(dataloader, model, loss_fn):
    #y_pred = []
    #y_true = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([])
    all_labels = all_labels.to(device)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            all_preds = torch.cat((all_preds, pred), dim=0)
            all_labels = torch.cat((all_labels, y), dim=0)
            #output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
            #label = y.data.cpu().numpy()
            #y_pred.extend(output)
            #y_true.extend(label)

            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print("TEST LOOP")
    print("    loss: ", test_loss)
    print("    accuracy: ", 100*correct)
    print("")
    #return y_pred, y_true
    return all_preds, all_labels, np.array([test_loss, correct])



def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

#> execution de l'apprentissage et des tests
for k in range(4): # itération sur 4 plis
    print(f"Fold {k+1}\n-------------------------------\n-------------------------------")
    model = Net()
    model = Net().to(device)
    model.to(torch.float)
    model.apply(reset_weights)

    #> fonction de perte et algorythme d'optimisation
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_sampler = torch.utils.data.ConcatDataset(( groups[(k+1)%4], groups[(k+2)%4], groups[(k+3)%4] ))
    train_dataloader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True, num_workers=0)
    
    valid_sampler = groups[k]
    
    '''
    for i in range(5):
        wished_idx = []
        for idx, label in enumerate(valid_sampler.datasets[i]):
            if label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]:
                wished_idx.append(idx)
            #indices = [idx for idx, target in enumerate(valid_sampler.datasets[i]) if target in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19]]
        valid_sampler.datasets[i] = torch.utils.data.Subset(valid_sampler.datasets[i], wished_idx)
    print(valid_sampler.datasets[i])
    '''

    valid_dataloader = DataLoader(valid_sampler, batch_size=batch_size, shuffle=True, num_workers=0)

    fold_perf = np.array([0,0,0])
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([])
    all_labels = all_labels.to(device)
    for t in range(epochs): # itération sur les epochs
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_train_perf = train_loop(train_dataloader, model, loss_fn, optimizer)
        fold_perf = np.vstack((fold_perf, np.insert(epoch_train_perf, 0, t)))
        valid_preds, valid_labels, epoch_valid_perf = valid_loop(valid_dataloader, model, loss_fn)
        fold_perf = np.vstack((fold_perf, np.insert(epoch_valid_perf, 0, t)))
        all_preds = torch.cat((all_preds, valid_preds), dim=0)
        all_labels = torch.cat((all_labels, valid_labels), dim=0)
    print("Done!")

    np.savetxt("./output/fold_perf"+str(k)+".csv", fold_perf, fmt='%1f', delimiter=';') #%1.5f

    torch.save(model.state_dict(), "./output/fold"+str(k)+".pt")

    stacked = torch.stack(
    (
        all_labels
        ,all_preds.argmax(dim=1)
    )
    ,dim=1
    )

    cmt = torch.zeros(23,23, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        tl = int(tl)
        pl = int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1
    

    np.set_printoptions(suppress=False)
    cmt = cmt.numpy()

    np.savetxt("./output/fold"+str(k)+".csv", cmt, fmt='%1.1d', delimiter=';')


    '''
    metric = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=23)
    metric.attach(default_evaluator, 'cm')
    #y_pred_tensor = torch.FloatTensor(y_pred_all)
    #y_true_tensor = torch.FloatTensor(y_true_all)
    y_pred = torch.stack(y_pred_all)
    y_pred = torch.FloatTensor(y_pred)
    print(y_pred)
    a = input("yolo")
    y_true = torch.stack(y_true_all)
    y_true = torch.FloatTensor(y_true)
    print(y_true)
    state = default_evaluator.run([[y_pred, y_true]])
    print(state.metrics['cm'])
    '''
    '''
    classes = {'trachea', 'right lung', 'left lung', 'pancreas', 'gallbladder', 'urinary bladder', 'sternum', 'first lumbar vertebra', 'right kidney', 'left kidney', 'right adrenal gland', 'left adrenal gland', 'right psoas major', 'left psoas major', 'muscle body of right rectus abdominis', 'muscle body of left rectus abdominis', 'aorta', 'liver', 'thyroid gland', 'spleen', 'background', 'body envelope', 'thorax-abdomen'}
    cf_matrix = confusion_matrix(y_true_all, y_pred_all)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    '''