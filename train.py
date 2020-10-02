import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

und = pd.read_csv("Undistorted.csv")
nat = pd.read_csv("Naturally-Blurred.csv")
art = pd.read_csv("Artificially-Blurred.csv")

df = pd.DataFrame()
df = df.append(und)
df = df.append(nat)
df = df.append(art)

X = df[["laplacian var","laplacian max","sobel mean","sobel var","sobel max","roberts mean","roberts var","roberts max"]].values
y = df["blur"].values

x_train,x_valid,y_train,y_valid = train_test_split(X,y,test_size=0.1)

svc_linear = SVC(C=100,kernel='linear')
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=10)

clf = VotingClassifier(estimators=[('svc_linear',svc_linear),('lr',lr),('rf',rf)], voting='hard')
clf.fit(x_train,y_train)

pred = clf.predict(x_valid)
print('Validation Accuracy:',accuracy_score(y_valid,pred))

with open('classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)