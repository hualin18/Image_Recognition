import pandas as pd
import numpy as np
import math

def tiidf_transformer(filename):
    path = 'data_folder/'
    df = pd.read_csv(path + filename)
    features = df[df.columns[0:-1]].values
    label = df[df.columns[-1]]
    m,n = features.shape
    tfidf_feature = np.copy(features).astype(float)
    for i in range(m):
        total = float(np.sum(features[i,:]))
        for j in range(n):
            term = float(features[i,j])
            nzero_doc = float(np.sum(features[:,j] != 0))
            try:
                tfidf_feature[i,j] = float(1000 * term / total * math.log(m/nzero_doc))
            except:
                print('zero!!!!!')

    df_tfidf = pd.DataFrame(tfidf_feature)

    pd.concat([df_tfidf, label], axis=1).to_csv('data_folder/tfidffeature' + filename, index=False, float_format='%g')
