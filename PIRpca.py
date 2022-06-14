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

def apply_pca(X, y):
    nbr_pca_features = 10  # nombre de composantes principales
    pca = PCA(n_components=nbr_pca_features)
    pca.fit(X, y)
    #print("X_train avant pca    ", X_train.dtype)
    # print(X_train.shape)
    X_train = pca.transform(X_train)
    #print("X_train apres pca sans transformation  ", X_train.dtype)
    X_train = X_train.astype('float16')
    return X_train