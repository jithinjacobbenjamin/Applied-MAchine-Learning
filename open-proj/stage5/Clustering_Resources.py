import pandas as pd
import operator
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
dataP = pd.read_csv("output_data\\CarbonfootprintresourcesReshaping.csv") #The data was imported
le=LabelEncoder()
for column in dataP.columns:
    dataP[column] = le.fit_transform(dataP[column].astype(str)) #Type of the data was changed
for column in dataP.columns:
    if dataP[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        dataP[column] = le.fit_transform(dataP[column])
dataP[column] = le.fit_transform(dataP[column].astype(str))
#dataCF = pd.read_csv("data\input_data\\Individual_Data.csv")
X = dataP.iloc[:, 0:4]
y = dataP.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50) #Testing and training the data
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)
from sklearn.cluster import KMeans #Import KMeans from sklearn
km=KMeans(n_clusters=5,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(X)
print(km.cluster_centers_)
print('SSE for Kmeans is= %3f' %km.inertia_)#Sum squared error was found
from sklearn.datasets import make_moons
X_moon,y_moon=make_moons(n_samples=200,noise=0.05,random_state=0) #makemoons
plt.scatter(X[:,0], X[:,1])
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward") #Hierarchial clustering was done
y_hc = hc.fit_predict(X)
print('Cluster labels = %s' %y_hc)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='pink', label="Cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label="Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='yellow', label="Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='red', label="Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='orange', label="Cluster 5")
plt.title("Hierarchial Clusters")
plt.xlabel("Species")
plt.ylabel("Value")
plt.legend()
plt.show()
reg = AgglomerativeClustering()
reg.fit(X, y)
y_hat = reg.fit_predict(X)
mse_lr=mean_squared_error(y, y_hat)
print("mse",mse_lr)
j=len(y_hc)
sse=j*mse_lr
print("sse using Agglomerative clustering is",sse)
from sklearn.cluster import DBSCAN #DBSCAN clustering was done
from sklearn.datasets import make_moons
X_moon,y_moon=make_moons(n_samples=200,noise=0.05,random_state=0)
db = DBSCAN( eps =0.2 , min_samples=5 , metric= 'euclidean')
y_db = db.fit_predict(X_moon)
print(y_db)
plt.scatter(X_moon[ y_db ==0 ,0],X_moon[ y_db ==0 ,1] ,c= 'pink')
plt.scatter(X_moon [ y_db ==1 ,0],X_moon [ y_db ==1 ,1] , c='magenta')
plt.show()
reg = DBSCAN()
reg.fit(X, y)
y_hat = reg.fit_predict(X) #fit_predict was conducted
mse_lr=mean_squared_error(y, y_hat)
print("mse",mse_lr)
j=len(y_hat)
sse=j*mse_lr
print("sse Using DBScan is",sse)
from sklearn.cluster import KMeans
wcss = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
print(type(X_train))
print(type(X_test))
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
print("SSE using Elbow method  is",kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
