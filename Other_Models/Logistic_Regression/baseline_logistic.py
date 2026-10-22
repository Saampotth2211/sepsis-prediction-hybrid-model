import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

BASE=r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DATA=os.path.join(BASE,"data","processed","lstm")

def features(d):

    X=d["X"]
    M=d["M"]

    feat=[]

    for i in range(len(X)):
        row=[]

        for j in range(40):

            vals=X[i,:,j]
            mask=M[i,:,j]==1

            if mask.sum()==0:
                row+= [0,0,0,0]
            else:
                v=vals[mask]
                row += [v.mean(),v.min(),v.max(),mask.sum()]

        feat.append(row)

    return np.array(feat)

train=np.load(os.path.join(DATA,"train_lstm.npz"))
val=np.load(os.path.join(DATA,"validation_lstm.npz"))
test=np.load(os.path.join(DATA,"test_lstm.npz"))

Xtr=features(train)
Xte=features(test)

clf=LogisticRegression(max_iter=500,class_weight="balanced")
clf.fit(Xtr,train["y"])

p=clf.predict_proba(Xte)[:,1]

pred=(p>=0.5).astype(int)

print("ROC:",roc_auc_score(test["y"],p))
print("PR :",average_precision_score(test["y"],p))
print("F1 :",f1_score(test["y"],pred))
print("Accuracy:",accuracy_score(test["y"],pred))