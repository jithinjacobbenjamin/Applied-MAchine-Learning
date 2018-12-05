import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df1=pd.read_csv('output_data\\Carbonfootprint_Reshaping.csv',header=None) #importing the dataset and reading it
le=LabelEncoder()
for column in df1.columns:
    df1[column] = le.fit_transform(df1[column].astype(str))
for column in df1.columns:
    if df1[column].dtype == type(object): #changing the datatype from object to float64
        le = preprocessing.LabelEncoder() #data preprocessing
        df1[column] = le.fit_transform(df1[column])
df1.head()
X=df1.loc[:,2:].values
y=df1.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)
le.classes_
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=2)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)
print('Test Accuracy using pipelining: %3f' % pipe_lr.score(X_test,y_test)) #Validation using pipelining
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import cross_val_score
kfold=StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)
scores=[]
for k,(train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])
    score=pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold: %2d,Class dist.: %s,Acc:%3f' % (k+1,np.bincount(y_train[train]),score)) #Validation using the KFolds
scores=cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV Accuracy scores using KFold: %s' % scores)
