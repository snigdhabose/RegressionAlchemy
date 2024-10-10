import pandas as pd
from numpy import linalg as LA
import numpy as np
df_train=pd.read_csv('D:/SnigdhaDocs/iitm/sem2/ml/Dataset2.csv',header=None)
ftr_trn = 0
MSlst=[]
w_grad_list=[]
X_train=df_train.iloc[:,:-1]
X_train=df_train.iloc[::,:-1]
for row_train in X_train:
    ftr_trn=ftr_trn +1
ftrain_w=ftr_trn
ntrain_w=len(df_train)

X_train=X_train.to_numpy()
y_train2=X_train.to_numpy()
for i in range(100):
    np.random.shuffle(X_train)
    X_train1=X_train[::100,::]
    y_train1=y_train2[::100,::]
    ftr_trn=0
    for row_train in X_train:
        ftr_trn=ftr_trn+1
    f_train=ftr_trn()
    n_train=len(X_train1)
    
    w_ml=np.dot(np.dot(LA.inv(np.dot(X_train1.T,X_train1)),X_train1.T),y_train1)
    
    MS=(1/n_train)*(LA.norm(X_train1@w_ml -y_train1))**2
    MSlst.append(MS)
    
    eta=0.001
    w_grad=np.random.randn(f_train,1)
    loss1=[]
    diff=[]
    
    for i in range(1100):
        w_grad=w_grad - eta*(X_train1.T@(X_train1@w_grad-y_train1))
        diff.append(LA.norm(w_grad - w_ml))
    w_grad_list.append(w_grad)
    diff=np.array(diff)
    diff
    
df_test=pd.read_csv('D:/SnigdhaDocs/iitm/sem2/ml/A2Q2Data_test.csv',header=None)
ftr_tst = 0
X_test1=df_test.iloc[:,:-1]
Y_test1=df_test.iloc[::,-1:]

for row_test in X_test1:
    ftr_tst = ftr_tst +1
f_test=ftr_tst()
n_test=len(df_test)

X_test1=X_test1.to_numpy()
y_test1=Y_test1.to_numpy()

print("Final Mean Squared Error on train set using w_ML",(1/n_train)*(LA.norm(X_train1@w_grad - y_train1))**2)
print("Final Mean Squared Error on train set using w_ML",(1/n_test)*(LA.norm(X_test1@w_grad - y_test1))**2)
