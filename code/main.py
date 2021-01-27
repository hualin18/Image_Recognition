import LRclassifier
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from nnclassfication import one_layer_nn, three_layer_nn
from torch import nn

if __name__ == "__main__":
    ## run Logistic on different dimension of Bow
    path = 'data_folder/countfeature/'
    tr_paths = [path+'train_feature20.csv',path+'train_feature50.csv',path+'train_feature100.csv',path+'train_feature200.csv', path+'train_feature500.csv',path+'train_feature1000.csv',path+'train_feature2000.csv']
    te_paths = [path+'test_feature20.csv',path+'test_feature50.csv',path+'test_feature100.csv',path+'test_feature200.csv', path+'test_feature500.csv',path+'test_feature1000.csv',path+'test_feature2000.csv']
    n_clusts = [20,50,100,200,500,1000,2000]

    proc = Pool(processes=7)
    proc.starmap(LRclassifier.LR_multi,zip(tr_paths,te_paths))
    proc.close()




    ## Tune C for Logistic with pelnalty l1 on features=200
    parameters = {'C': np.concatenate((np.arange(0.1,1.9,0.2), np.arange(2, 5, 0.6), np.arange(5, 21, 2)))}
    LRclassifier.LR_multi_tune(tr_paths[3],te_paths[3],parameters)


    #  nural network classification
    n_epochs = [1, 5, 10, 20, 50, 100]

    for n_epoch in n_epochs:
        for tr_path, te_path, n in zip(tr_paths, te_paths, n_clusts):
            one_layer_nn(tr_path, te_path, inputdim=n, num_epochs=n_epoch)
            three_layer_nn(tr_path, te_path, inputdim=n, num_epochs=n_epoch)

    # Because n_epochs = 5 shows a great performance and c_clustes=500 is the best, tune Func in 3layers
    for i in range(5):
        three_layer_nn(tr_paths[4], te_paths[4], inputdim=500, num_epochs=5, Func=nn.Sigmoid(),
                       store_path='data_folder/sigmoid_' + str(i) + '_trial_')
        three_layer_nn(tr_paths[4], te_paths[4], inputdim=500, num_epochs=5, Func=nn.Tanh(),
                       store_path='data_folder/tanh_' + str(i) + '_trial_')
        three_layer_nn(tr_paths[4], te_paths[4], inputdim=500, num_epochs=5, Func=nn.ReLU(),
                       store_path='data_folder/relu_' + str(i) + '_trial_')