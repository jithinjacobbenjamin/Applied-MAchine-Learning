from sklearn import tree
from sklearn.svm import SVC
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder #LabelEncoder for preprocessing
from sklearn.neighbors import KNeighborsClassifier

import  numpy as np
le=LabelEncoder()
dataP=pd.read_csv('output_data\\Carbonfootprint_Reshaping.csv',header=None) #Data is loaded
for column in dataP.columns:
    dataP[column] = le.fit_transform(dataP[column].astype(str)) #Type conversion is done to float64
for column in dataP.columns:
    if dataP[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        dataP[column] = le.fit_transform(dataP[column])
dataP[column] = le.fit_transform(dataP[column].astype(str))

from sklearn import datasets
X = dataP.iloc[:,[0,1]]
Y = dataP.iloc[:,1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.5) #Training and testing data
clf_tree = tree.DecisionTreeClassifier() # Decision tree model imported from sklearn library
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
print ('Best classifier is {}'.format(classifiers[index])) #Checks which is the best classifier based on highest accuracy