import pandas as pd
import operator
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder #LabelEncoder for preprocessing of data
from sklearn.metrics import mean_squared_error
dataP = pd.read_csv("output_data\\CarbonfootprintresourcesReshaping.csv") #import the dataset
le=LabelEncoder()
for column in dataP.columns:
    dataP[column] = le.fit_transform(dataP[column].astype(str)) #Type conversion to float64 is done here
for column in dataP.columns:
    if dataP[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        dataP[column] = le.fit_transform(dataP[column])
dataP[column] = le.fit_transform(dataP[column].astype(str))
#dataCF = pd.read_csv("data\input_data\\Individual_Data.csv")
X = dataP.iloc[:, 0:4]
y = dataP.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.5)
clf_tree = tree.DecisionTreeClassifier()# Tree model imported from sklearn library
clf_svm = SVC()
clf_perceptron = Perceptron()# Perceptron model imported from sklearn library
clf_KNN = KNeighborsClassifier()
clf_tree.fit(X_train,Y_train)
clf_svm.fit(X_train,Y_train)
clf_perceptron.fit(X_train,Y_train)
clf_KNN.fit(X_train,Y_train)
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(Y_test, pred_tree) * 100
print ('Accuracy for tree: {}' .format(acc_tree))
pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, pred_svm) * 100
print ('Accuracy for svm: {}' .format(acc_svm))
pred_perceptron = clf_perceptron.predict(X_test)
acc_perceptron = accuracy_score(Y_test, pred_perceptron) * 100
print ('Accuracy for perceptron: {}' .format(acc_perceptron))
pred_KNN = clf_KNN.predict(X_test)
acc_KNN = accuracy_score(Y_test, pred_KNN) * 100
print ('Accuracy for KNN: {}' .format(acc_KNN))
index = np.argmax([acc_svm,acc_perceptron,acc_KNN,acc_tree])
classifiers = {0:'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'Tree'}
print ('Best classifier is {}'.format(classifiers[index])) #The best classifier is chosen