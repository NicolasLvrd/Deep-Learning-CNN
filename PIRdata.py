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

svmclassifier = SVC(kernel="rbf", C=1000)
'''
n_estimators = 10
svmclassifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', C=1, probability=False, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
'''

total_avg_accuracy = np.array([0,0,0]) # [fold number ; train acc ; test acc]
total_avg_recall = np.array([0,0,0]) # [fold number ; train rec ; test rec]
for i in range(NbrPlis):
    print("FOLD ", i)
    TrainingDatas = []  # stock the training datas for each folder
    TestingDatas = []  # stock the testing datas for each folder

    fold_avg_accuracy = np.array([0,0])
    fold_avg_recall = np.array([0,0])
    fold_organs_accuracy = np.array([0,0])

    # -------------------- TRAINING -------------------- #
    X_train = np.load("X_train_fold"+str(i)+".npy")
    y_train = np.load("y_train_fold"+str(i)+".npy")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    t1_start = perf_counter()
    svmclassifier.fit(X_train, y_train)
    t1_stop = perf_counter()
    print("")
    print("Elapsed time during the training process for the plis n°", i+1, "in seconds:", t1_stop-t1_start)

    y_pred_train = svmclassifier.predict(X_train)

    print("")
    print(
        f"The model is {accuracy_score(y_pred_train,y_train)*100}% accurate sur la base de train pour le pli n° {i+1}")

    totalTrainingAccuracy += accuracy_score(y_pred_train, y_train)
    foldTrainingAccuracy = accuracy_score(y_pred_train, y_train)
    fold_avg_accuracy[0] = foldTrainingAccuracy
    
    joblib.dump(svmclassifier, 'svmPliN'+str(i+1)+'.pkl')
    print("le classifieur a bien été sauvegardé")

    # vider les array
    train = []
    X_train = []
    y_train = []
    y_pred_train = []

    # -------------------- TESTING -------------------- #
    X_test = np.load("X_test_fold"+str(i)+".npy")
    y_test = np.load("y_test_fold"+str(i)+".npy")
    # predicting classes
    y_pred_test = svmclassifier.predict(X_test)

    # print("")
    # print(
    # f"The model is {accuracy_score(y_pred_test,y_test)*100}% accurate sur base de test pour le pli n° {i+1}")

    totalTestingAccuracy += accuracy_score(y_pred_test, y_test)
    foldTestingAccuracy = accuracy_score(y_pred_test, y_test)
    fold_avg_accuracy[1] = foldTestingAccuracy
    print("Testing Accuracy:", totalTestingAccuracy)

    # -------------------- SAVE FOLD ACCURACY -------------------- #
    total_avg_accuracy = np.vstack((total_avg_accuracy, np.insert(fold_avg_accuracy, 0, i)))
    np.savetxt("./output_SVM/fold_accuracy"+str(i)+".csv", total_avg_accuracy, fmt='%1f', delimiter=';')

    # -------------------- PLOT CONFUSION MATRIX -------------------- #
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

np.savetxt("./output_SVM/organs_accuracy"+str(i)+".csv", précisionParOrganes, fmt='%1f', delimiter=';')

totalTrainingAccuracy = (totalTrainingAccuracy/NbrPlis)*100
totalTestingAccuracy = (totalTestingAccuracy/NbrPlis)*100
print("Tous les folds terminés \n")
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
