import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

## get the filenames for image
data = pd.read_csv('data_folder/newdata.csv')
ids = data['id'].values
y = data['landmark_id'].values


## apply sift to all images
def get_features(filename):
    path = '/home/huachao/Documents/141_project/train/'
    img = cv2.imread(path + filename + '.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des
dess = []
labels = []
for i, id in enumerate(ids):
    des = get_features(id)
    if des is not None:
        dess.append(des)
        labels.append(y[i])

# chang lable as 0,1,2...
y = np.zeros(len(labels))
for i,c in enumerate(np.unique(labels)):
    y += (i*(labels==c))
y = y.astype(int)

# split train test
feature_tr, feature_te, y_tr, y_te = train_test_split(dess,y,test_size=0.25,random_state=16)

## Define function to get bag of words
def img_bow(feature_tr,feature_te,n_cluster):
    model = MiniBatchKMeans(n_clusters= n_cluster)
    tr_descriptors = np.array([des for dess_list in feature_tr for des in dess_list])
    # train kmeans on all descriptors
    model.fit(tr_descriptors)
    # compute cluster words from features of each image
    img_words_tr = [model.predict(dess_list) for dess_list in feature_tr]
    img_words_te = [model.predict(dess_list) for dess_list in feature_te]
    # word counts for each image.
    return np.array([np.bincount(words, minlength=n_cluster) for words in img_words_tr]), np.array([np.bincount(words, minlength=n_cluster) for words in img_words_te])

## calculate BOW with different n_clusters in kmeans
n_clust = [20,50,100,200,500,1000,2000]
for n in n_clust:
    tr, te = img_bow(feature_tr,feature_te,n)
    pd.concat([pd.DataFrame(tr),pd.DataFrame(y_tr)],axis=1).to_csv('data_folder/countfeature/train_feature' + str(n) + '.csv',index=False)
    pd.concat([pd.DataFrame(te), pd.DataFrame(y_te)], axis=1).to_csv('data_folder/countfeature/test_feature' + str(n) + '.csv',index=False)