import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals.joblib import parallel_backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")
## Specific Logistic regression
def LR_multi(tr_path,te_path):
    # multinomial Logistic
    df_tr = pd.read_csv(tr_path)
    df_te = pd.read_csv(te_path)
    features_tr = df_tr[df_tr.columns[0:-1]].values
    y_tr = df_tr[df_tr.columns[-1]].values
    features_te = df_te[df_te.columns[0:-1]].values
    y_te = df_te[df_te.columns[-1]].values
    clf = LogisticRegression(random_state=42,multi_class='multinomial',solver='saga',max_iter=10000)
    clf.fit(features_tr,y_tr)
    ##  score and accuracy
    y_pred = clf.predict(features_te)
    n = features_tr.shape[1]
    pd.DataFrame(y_pred).to_csv('data_folder/lyp'+ str(n)+ '.csv', index=False)


def LR_multi_tune(tr_path,te_path,parameters):
    # multinomial Logistic
    df_tr = pd.read_csv(tr_path)
    df_te = pd.read_csv(te_path)
    features_tr = df_tr[df_tr.columns[0:-1]].values
    y_tr = df_tr[df_tr.columns[-1]].values
    features_te = df_te[df_te.columns[0:-1]].values
    y_te = df_te[df_te.columns[-1]].values

    clf = LogisticRegression(penalty='l1',random_state=42,multi_class='multinomial',solver='saga',max_iter=10000)
    ## choose the best parameter by python itself x
    gridclf = GridSearchCV(clf,parameters,scoring='r2')


    with parallel_backend('threading',n_jobs=20):
        gridclf.fit(features_tr,y_tr)
    ##  score and accuracy
    y_pred = gridclf.predict(features_te)
    print(gridclf.best_estimator_,gridclf.best_estimator_)
    n = features_tr.shape[1]
    pd.DataFrame(y_pred).to_csv('data_folder/lyp_bestC' + str(n) + '.csv', index=False)


