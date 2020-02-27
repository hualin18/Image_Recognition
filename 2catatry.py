## select the tareget image

import pandas as pd
import numpy as np

dataset = pd.read_csv('/home/huachao/Documents/141_project/train.csv')
cata, count = np.unique(dataset['landmark_id'],return_counts=True)
cc = np.vstack((count,cata))
cc_sort = cc[:,cc[0].argsort()]
print(cc[0])
print(cc[1])
print(cc_sort[0])
print(cc_sort[1])
print(dataset.head())
def dataset_create(i):
    id_to_keep = cc_sort[1,-i:]
    mydata = [dataset[(dataset['landmark_id'] == id)] for id in id_to_keep]
    df = pd.concat(mydata)[['id','landmark_id']]
    df.to_csv('data_folder/newdata.csv',index=False)

dataset_create(20)