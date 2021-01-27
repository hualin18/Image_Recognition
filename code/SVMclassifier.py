import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
## Get the features
def SVM(path):
    df=pd.read_csv(path)
    features = df[df.columns[0:-1]].values
    label = df[df.columns[-1]].values

    D_tr, D_te, y_tr, y_te = train_test_split(features, label, train_size=0.7, random_state=123)
    print(D_tr.shape, D_te.shape, y_tr.shape, y_te.shape)  ##check data
    clf = svm.SVC(random_state=42)

    parameters = {'gamma': np.power(10, np.arange(-3, 5, 0.5)),'kernel':['linear','rbf','poly','sigmoid']}
    ## choose the best parameter by python itself x
    gridclf = GridSearchCV(clf,parameters,scoring='r2')
    gridclf.fit(D_tr,y_tr)
    ##  score and accuracy
    y_pred = gridclf.predict(D_te)
    accuracy = np.sum(y_pred == y_te) / y_te.shape[0]
    return accuracy
