import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df2=pd.read_csv('output_data\\Carbonfootprint_Reshaping.csv',header=None) #importing the dataset

for column in df2.columns:
    if df2[column].dtype == type(object):
        le = preprocessing.LabelEncoder() #Data preprocessing was done to handle the dataset
        df2[column] = le.fit_transform(df2[column])
df2.head()
X1=df2.loc[:,2:].values
y1=df2.loc[:,1].values
le=LabelEncoder()
y1=le.fit_transform(y1)
le.classes_
X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.20,random_state=2) #Training and testing of data was done
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
hc = AgglomerativeClustering(n_clusters=7, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X1)
print('Cluster labels = %s' %y_hc)
plt.scatter(X1[y_hc == 0, 0], X1[y_hc == 0, 1], s=100, c='pink', label="Cluster 1")
plt.scatter(X1[y_hc == 1, 0], X1[y_hc == 1, 1], s=100, c='blue', label="Cluster 2")
plt.scatter(X1[y_hc == 2, 0], X1[y_hc == 2, 1], s=100, c='yellow', label="Cluster 3")
plt.scatter(X1[y_hc == 3, 0], X1[y_hc == 3, 1], s=100, c='red', label="Cluster 4")
plt.scatter(X1[y_hc == 4, 0], X1[y_hc == 4, 1], s=100, c='orange', label="Cluster 5")
plt.scatter(X1[y_hc == 5, 0], X1[y_hc == 5, 1], s=100, c='black', label="Cluster 6")
plt.scatter(X1[y_hc == 6, 0], X1[y_hc == 6, 1], s=100, c='purple', label="Cluster 7") #Clusters were plotted
plt.title("Hierarchial Clusters")
plt.xlabel("Species")
plt.ylabel("Value")
plt.legend()
plt.show()
reg = AgglomerativeClustering()# Hierarchial clustering imported from sklearn library
reg.fit(X1, y1)
y_hat = reg.fit_predict(X1)
mse_lr=mean_squared_error(y1, y_hat)
print("mse",mse_lr)
j=len(y_hc)
sse=j*mse_lr
print("sse using agglomerative clustering",sse)
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
X_moon,y_moon=make_moons(n_samples=200,noise=0.05,random_state=0)
db = DBSCAN( eps =0.2 , min_samples=5 , metric= 'euclidean')
y_db = db.fit_predict(X_moon)
print(y_db)
plt.scatter(X_moon[ y_db ==0 ,0],X_moon[ y_db ==0 ,1] ,c= 'pink')
plt.scatter(X_moon [ y_db ==1 ,0],X_moon [ y_db ==1 ,1] , c='magenta')
plt.show()
reg = DBSCAN()
reg.fit(X1, y1)
y_hat = reg.fit_predict(X1) #fit_predict was done
mse_lr=mean_squared_error(y1, y_hat)
print("mse",mse_lr)
j=len(y_hat)
sse=j*mse_lr
print("sse using dbscan",sse)
from sklearn.cluster import KMeans # Kmeans for elbow imported from sklearn library
wcss = []
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=21)
print(type(X_train))
print(type(X_test))
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
print("SSE using elbow is",kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()