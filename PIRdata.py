# import all the librairies
#!/usr/bin/python

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
# training the algorithm

# C = 10  # SVM regularization parameter
# gamma = 0.0001
# svclassifier = SVC(kernel="rbf", C=C, gamma= gamma)
# svclassifier = LinearSVC(random_state=0)
# svclassifier = LinearSVC(random_state=0)
# svclassifier = LinearSVC(C=0.0005, random_state=13)
# svclassifier = SVC(kernel="linear", max_iter = 3)
# svclassifier = LinearSVC(C=0.0005,dual=False)
# svmclassifier = LinearSVC(C=1, dual=False, class_weight='balanced')

# dual = False prefered when nbr samples > nbr features
# C = regularization parameter to avoid overfitting
# random_state has no effect on the results if dual = False
# max_iter = maximum number of iterations to perform

# import the dataseet + concatener data set

# Open a file
path = "data\CTce_ThAb_b33x33_n1000_8bit\\"


# met les noms de fichier contenu dans ce dossier (path) dans un tableau dirs
dirs = os.listdir(path)
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

#array = array.astype('float16')
# print(array)
AllData = np.concatenate(array[:Nbrfile])
#AllData = np.round(AllData)
AllData = AllData.astype('float16')

# C = 0.001  # SVM regularization parameter
# svmclassifier = SVC(kernel="rbf", C=1)
n_estimators = 10
svmclassifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', C=1, probability=False, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
# totalTrainingAccuracy = 0
# svmclassifier = LinearSVC(
# C=0.0001, dual=False, class_weight='balanced', max_iter=1500)

for i in range(NbrPlis):
    TrainingDatas = []  # stock the training datas for each folder
    TestingDatas = []  # stock the testing datas for each folder

    train = np.concatenate(
        array[0:nbFichiersTestParPlis*i]+array[nbFichiersTestParPlis*(i+1):Nbrfile])
    train = train.astype('float16')
    # [x_train,y_train]

    #TrainingDatas.append([train[:, 1:], np.ravel(train[:, :1])])

    X_train = train[:, 1:]
    y_train = np.ravel(train[:, :1])
    print("")
    print("DATA de Train pour le pli", i+1, " : OK")

    # pca
    nbr_pca_features = 10  # nombre de composantes principales
    #pca = PCA(n_components=nbr_pca_features)
    #pca.fit(X_train, y_train)

    #print("X_train avant pca    ", X_train.dtype)
    # print(X_train.shape)

    #X_train = pca.transform(X_train)

    #print("X_train apres pca sans transformation  ", X_train.dtype)
    X_train = X_train.astype('float16')
    #print("X_train apres pca avec transformation  ", X_train.dtype)
    print("")
    print("la pca sur X_train a bien été éffectuée")

    t1_start = perf_counter()
    svmclassifier.fit(X_train, y_train)
    t1_stop = perf_counter()
    print("")
    print("Elapsed time during the training process for the plis n°",
          i+1, "in seconds:", t1_stop-t1_start)

    y_pred_train = svmclassifier.predict(X_train)

    print("")
    print(
        f"The model is {accuracy_score(y_pred_train,y_train)*100}% accurate sur la base de train pour le pli n° {i+1}")

    totalTrainingAccuracy += accuracy_score(y_pred_train, y_train)

    joblib.dump(
        pca, 'pcaPliN'+str(i+1)+'.pkl')
    print("le pca a bien été sauvegardé")

    joblib.dump(
        svmclassifier, 'svmPliN'+str(i+1)+'.pkl')
    print("le classifieur a bien été sauvegardé")

    # vider les array
    train = []
    X_train = []
    y_train = []
    y_pred_train = []

    test = np.concatenate(
        array[nbFichiersTestParPlis*i:nbFichiersTestParPlis*(i+1)])
    test = test.astype('float16')
    # [x_test,y_test]
    #TestingDatas.append([test[:, 1:], np.ravel(test[:, :1])])

    X_test = test[:, 1:]

    y_test = np.ravel(test[:, :1])

    # print("")
    #print("DATA de Test pour le pli", i+1, " : OK")

    # predicting classes
    y_pred_test = svmclassifier.predict(pca.transform(X_test))

    # print("")
    # print(
    # f"The model is {accuracy_score(y_pred_test,y_test)*100}% accurate sur base de test pour le pli n° {i+1}")

    totalTestingAccuracy += accuracy_score(y_pred_test, y_test)

    ConfMatr = confusion_matrix(y_pred_test, y_test)

    (l, a) = ConfMatr.shape
    for i in range(l):
        for j in range(a):
            matConfCumulee[i][j] += ConfMatr[i][j]
    for i in range(l):
        totalprecision = 0
        totalRecall = 0
        for j in range(l):
            totalprecision += ConfMatr[j][i]
            totalRecall += ConfMatr[i][j]
            if (totalprecision != 0):
                précisionParOrganes[i] += ConfMatr[i][i]/totalprecision
            if (totalRecall != 0):
                recallParOrganes[i] += ConfMatr[i][i]/totalRecall

    # vider les array
    X_train = []
    y_train = []
    y_pred_train = []
    test = []
    ConfMatr = []

totalTrainingAccuracy = (totalTrainingAccuracy/NbrPlis)*100
totalTestingAccuracy = (totalTestingAccuracy/NbrPlis)*100
print("")
print("la moyenne de l'accuracy sur la base de train est de : ", totalTrainingAccuracy)
print("la moyenne de l'accuracy sur la base de test est de : ", totalTestingAccuracy)

organs = {
    1247: "Trachea",
    1302: "Right Lung",
    1326: "Left Lung",
    170: "Pancreas",
    187: "Gallbladder",
    237: "Urinary Bladder",
    2473: "Sternum",
    29193: "First Lumbar Vertebra",
    29662: "Right Kidney",
    29663: "Left Kidney",
    30324: "Right Adrenal Gland",
    30325: "Left Adrenal Gland",
    32248: "Right Psoas Major",
    32249: "Left Psoas Major",
    40357: "Right rectus abdominis",
    40358: "Left rectus abdominis",
    480: "Aorta",
    58: "Liver",
    7578: "Thyroid Gland",
    86: "Spleen",
    0: "Background",
    1: "Body Envelope",
    2: "Thorax-Abdomen"
}

lo = [organs[k] for k in sorted(organs.keys())]

prec_recall = []
for i in range(len(lo)):
    prec_recall.append(lo[i]+' : '+str(round(précisionParOrganes[i]/NbrPlis, 3)
                                       )+' : '+str(round(recallParOrganes[i]/NbrPlis, 3))+'\n')
print("")
print("")
print("précision par organes : ", prec_recall)


plt.figure(figsize=(25, 25))
plt.imshow(matConfCumulee, cmap='rainbow_r')
plt.title("Confusion Matrix for test Data", fontsize=20)
plt.xticks(np.arange(23), lo, rotation=90)
plt.yticks(np.arange(23), lo)
plt.ylabel('Actual Label', fontsize=15)
plt.xlabel('Predicted Label', fontsize=15)
plt.colorbar()
width, height = matConfCumulee.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(matConfCumulee[x][y]), xy=(
            y, x), horizontalalignment='center', verticalalignment='center')

plt.savefig("MatConf.png")
matConfCumulee = matConfCumulee[3:, 3:]

t0_stop = perf_counter()
print("Le programme a run en : ", t0_stop-t0_start, "secondes")

plt.show()

# print("précision finale = ", totalTrainingAccuracy)
