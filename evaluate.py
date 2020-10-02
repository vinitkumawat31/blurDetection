import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

dig_eval = pd.read_csv("DigitalBlurSet.csv")
nat_eval = pd.read_csv("NaturalBlurSet.csv")

X_dig_eval = dig_eval[["laplacian var","laplacian max","sobel mean","sobel var","sobel max","roberts mean","roberts var","roberts max"]].values
y_dig_eval = clf.predict(X_dig_eval)
df_res_dig = pd.DataFrame({"name":dig_eval["name"].values,"blur":y_dig_eval})

X_nat_eval = nat_eval[["laplacian var","laplacian max","sobel mean","sobel var","sobel max","roberts mean","roberts var","roberts max"]].values
y_nat_eval = clf.predict(X_nat_eval)
df_res_nat = pd.DataFrame({"name":nat_eval["name"].values,"blur":y_nat_eval})

df_nat = pd.read_excel("NaturalBlurSet.xlsx").sort_values(by=['Image Name'])
acc_nat = accuracy_score(df_nat["Blur Label"].values==1,df_res_nat["blur"].values==1.0)
print('Evaluation Accuracy on NaturalBlurSet:',acc_nat)

df_dig = pd.read_excel("DigitalBlurSet.xlsx").sort_values(by=['MyDigital Blur'])
acc_dig = accuracy_score(df_dig["Unnamed: 1"].values==1,df_res_dig["blur"].values==1.0)
print('Evaluation Accuracy on DigitalBlurSet:',acc_dig)

acc_eval = (acc_nat*df_nat.shape[0]+acc_dig*df_dig.shape[0])/(df_nat.shape[0]+df_dig.shape[0])
print('Total Evaluation Accuracy:',acc_eval)