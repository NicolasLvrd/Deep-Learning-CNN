import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from time import perf_counter
import joblib

path = "data\CTce_ThAb_b33x33_n1000_8bit\\"

array = []

Nbrfile = 20
NbrPlis = 4
nbFichiersTestParPlis = int(Nbrfile/NbrPlis)
matConfCumulee = np.zeros((23, 23))
précisionParOrganes = [0.0]*23  # 23 est le nombre de caractéristiques
recallParOrganes = [0.0]*23  # 23 est le nombre de caractéristiques
totalTrainingAccuracy = 0
totalTestingAccuracy = 0

t0_start = perf_counter()

for i in range(Nbrfile):  # pour chaque fichier
    with open(path + dirs[i]) as file_name:  # ouvre le fichier
        # array conytenant les data du fichier csv i qui s'ajoute pour chaque fichiers
        array.append(np.loadtxt(file_name, delimiter=","))
    print(str(i+1)+" fichiers extraits")

train = np.concatenate(
    array[0:nbFichiersTestParPlis*i]+array[nbFichiersTestParPlis*(i+1):Nbrfile])
train = train.astype('float16')

X_train = train[:, 1:]
y_train = np.ravel(train[:, :1])

nbr_pca_features = 10  # nombre de composantes principales
pca = PCA(n_components=nbr_pca_features)
pca.fit(X_train, y_train)
X_train = pca.transform(X_train)
X_train = X_train.astype('float16')
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)