import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sklearn
import sys

k = int(sys.argv[1]) # fold number

#> hyperparamètres
learning_rate = 1e-4
batch_size = 4
epochs = 1

# régularisation L1, par Szymon Maszke
# https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
class L1(torch.nn.Module):
    def __init__(self, module, weight_decay=0.00001):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)

# utilisation du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

'''
train_groups = [] # datasets regroupés par groupe de 5 patients pour le 4-folds
test_groups = []

patients = [] # datasets de chaque patient
for i in range(20):
    patients.append( torch.load("./data/train_patient"+str(i), map_location=torch.device('cpu')) )


for i in range(4):
    train_groups.append((patients[5*i], patients[5*i+1], patients[5*i+2], patients[5*i+3], patients[5*i+4])) #torch.utils.data.ConcatDataset(

patients = [] # datasets de chaque patient
for i in range(20):
    patients.append( torch.load("./data/test_patient"+str(i), map_location=torch.device('cpu')) )

for i in range(4):
    test_groups.append((patients[5*i], patients[5*i+1], patients[5*i+2], patients[5*i+3], patients[5*i+4]))
'''

train_labels_list = [None]*4
test_labels_list = [None]*4
train_images_list = [None]*4
test_images_list = [None]*4

for i in range(4):
    train_labels_list[i] = torch.load("./data/train/labels/group"+str(i), map_location=torch.device('cpu'))
    train_images_list[i] = torch.load("./data/train/images/group"+str(i), map_location=torch.device('cpu'))
    test_labels_list[i] = torch.load("./data/test/labels/group"+str(i), map_location=torch.device('cpu'))
    test_images_list[i] = torch.load("./data/test/images/group"+str(i), map_location=torch.device('cpu'))

#> définition du réseau
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5) # 10 x (33-6+1) x 28
        self.pool = nn.MaxPool2d(2) # 10 x 7 x 7
        self.conv2 = nn.Conv2d(6, 16, 5) # 20 x (7-6+1) x 2
        self.fc1 = nn.Linear(400, 150)
        self.fc2 = nn.Linear(150, 46)
        self.fc3 = nn.Linear(150, 23)
        self.dropout = nn.Dropout(0.25)
        '''
        
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.conv3 = nn.Conv2d(40, 60, 2)
        self.fc1 = nn.Linear(240, 180)
        self.fc2 = nn.Linear(180, 75)
        self.fc3 = nn.Linear(75, 23)
        '''
        '''
        self.conv1 = nn.Conv2d(1, 6, 7)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.conv3 = nn.Conv2d(40, 15, 2)
        self.fc1 = nn.Linear(144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 23)
        '''

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print("forward: ", x.shape)
        return x

#> boucle d'apprentissage
def train_loop(dataloader, model, loss_fn, optimizer):
    #dataiter = iter(dataloader)
    #images, labels = dataiter.next()
    #img = images[0][0]
    #img = img.cpu()
    #npimg = img.numpy()
    #plt.imshow(npimg)
    #plt.show()

    size = len(dataloader.dataset)
    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        #print("    train X: ", X.shape)
        #print("    train pred: ", pred)
        #print("    train pred argmax: ", pred.argmax(1))
        #print("    train pred argmax == y: ", pred.argmax(1) == y)
        #print("    train y: ", y)
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
    return np.array([train_loss, correct]) # pour écriture dans csv

#> boucle de validation
def valid_loop(dataloader, model, loss_fn):
    #y_pred = []
    #y_true = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    all_preds = torch.tensor([]).to('cpu')
    all_labels = torch.tensor([]).to('cpu')

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            soft = nn.Softmax(dim=0)
            all_preds = torch.cat((all_preds, soft(pred).argmax(1).detach()), dim=0)
            all_labels = torch.cat((all_labels, y.detach()), dim=0)

            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print("    VALID pred: ", pred)
            print("    VALID y: ", y)
            #a = input("break")
            # print("pred: ", pred)
            # print("pred.argmax(1): ", pred.argmax(1))
            # print("pred.argmax(1) == y: ", (pred.argmax(1) == y))
            # print("sum: ", (pred.argmax(1) == y).type(torch.float).sum())
    test_loss /= num_batches
    correct /= size
    print("TEST LOOP")
    print("    loss: ", test_loss)
    print("    accuracy: ", 100*correct)
    print("")
    return all_preds, all_labels, np.array([test_loss, correct])

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

print(f"Fold {k+1}\n-------------------------------\n-------------------------------")
model = Net()
model = Net().to(device)
model.to(torch.float)
model.apply(reset_weights)

#> fonction de perte et algorythme d'optimisation
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=0) #, betas=(0.90, 0.999), weight_decay=0.0099
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #weight_decay=1e-5

#train_sampler = torch.utils.data.ConcatDataset(( train_groups[(k+1)%4], train_groups[(k+2)%4], train_groups[(k+3)%4] ))
train_images = torch.cat( ( train_images_list[(k+1)%4],  train_images_list[(k+2)%4], train_images_list[(k+3)%4]), dim=0 ).to(device)
train_labels = torch.cat( ( train_labels_list[(k+1)%4],  train_labels_list[(k+2)%4], train_labels_list[(k+3)%4]) ).to(device) 
print(torch.unique(train_labels))
#a = input("break")
train_sampler = TensorDataset(train_images, train_labels)
train_dataloader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True, num_workers=0)

valid_sampler = TensorDataset(test_images_list[k].to(device), test_labels_list[k].to(device))
valid_dataloader = DataLoader(valid_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

fold_perf = np.array([0,0,0,0,0])
all_preds = torch.tensor([])
all_labels = torch.tensor([])
for t in range(epochs): # itération sur les epochs
    print(f"Epoch {t+1}\n-------------------------------")
    epoch_train_perf = train_loop(train_dataloader, model, loss_fn, optimizer)
    valid_preds, valid_labels, epoch_valid_perf = valid_loop(valid_dataloader, model, loss_fn)
    epoch_perf = np.concatenate((epoch_train_perf, epoch_valid_perf))
    fold_perf = np.vstack((fold_perf, np.insert(epoch_perf, 0, t)))
    all_preds = torch.cat((all_preds, valid_preds), dim=0)
    all_labels = torch.cat((all_labels, valid_labels), dim=0)
    # fold_perf = np.vstack((fold_perf, np.insert(epoch_valid_perf, 0, t)))
    # fold_perf = np.vstack((fold_perf, np.insert(epoch_train_perf, 0, t)))
print("Done!")

np.savetxt("./output/fold_perf"+str(k)+".csv", fold_perf, fmt='%1f', delimiter=';') #%1.5f

torch.save(model.state_dict(), "./output/fold"+str(k)+".pt")

organs = {
            1247 : "Trachea",
            1302 : "Right Lung",
            1326 : "Left Lung",
            170 : "Pancreas",
            187 : "Gallbladder",
            237 : "Urinary Bladder",
            2473 : "Sternum",
            29193 : "First Lumbar Vertebra",
            29662 : "Right Kidney",
            29663 : "Left Kidney",
            30324 : "Right Adrenal Gland",
            30325 : "Left Adrenal Gland",
            32248 : "Right Psoas Major",
            32249 : "Left Psoas Major",
            40357 : "Right rectus abdominis",
            40358 : "Left rectus abdominis",
            480 : "Aorta",
            58 : "Liver",
            7578 : "Thyroid Gland",
            86 : "Spleen",
            0 : "Background",
            1 : "Body Envelope",
            2 : "Thorax-Abdomen"
        }

lo = sorted([organs[i] for i in organs])

confusionMX = sklearn.metrics.confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,10))
plt.imshow(confusionMX,cmap='rainbow_r')
plt.title("Confusion Matrix for test Data of fold number "+str(i+1), fontsize=20)
plt.xticks(np.arange(23),lo, rotation=90)
plt.yticks(np.arange(23),lo)
plt.ylabel('Actual Label', fontsize=15)
plt.xlabel('Predicted Label', fontsize=15)
plt.colorbar()
width,height = confusionMX.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(confusionMX[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
plt.savefig("CMX_fold_"+str(k)+".png", bbox_inches='tight', dpi=300)

''' MATRICE DE CONFUSION FAIT MAISON
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

''' MATRICE DE CONFUSION PYTORCH
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