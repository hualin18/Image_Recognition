import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns


# Evalate one layer and three layer nn

def get_acc_onelay(head,tail):
    direct = 'data_folder/' + str(head) +'epochs_onelay' + str(tail) + '.csv'
    df = pd.read_csv(direct)
    matches = df[df['0']==df['0.1']]
    acc_rate = len(matches)/len(df)
    return acc_rate

def get_acc_threelay(head,tail):
    direct = 'data_folder/' + str(head) +'epochs_threelay' + str(tail) + '.csv'
    df = pd.read_csv(direct)
    matches = df[df['0']==df['0.1']]
    acc_rate = len(matches)/len(df)
    return acc_rate


one_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(1,i)
    one_epoches_one_layer.append(acc)

# get 1 epoaches one layer with all the k means
one_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(1,i)
    one_epoches_three_layer.append(acc)

# get 5 epoaches one layer with all the k means
five_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(5,i)
    five_epoches_one_layer.append(acc)
# get 5 epoaches three layer with all the k means
five_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(5,i)
    five_epoches_three_layer.append(acc)

# get 10 epoaches one layer with all the k means
ten_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(10,i)
    ten_epoches_one_layer.append(acc)
# get 10 epoaches three layer with all the k means
ten_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(10,i)
    ten_epoches_three_layer.append(acc)
# get 20 epoaches one layer with all the k means
twenty_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(20,i)
    twenty_epoches_one_layer.append(acc)
# get 20 epoaches three layer with all the k means
twenty_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(20,i)
    twenty_epoches_three_layer.append(acc)

# get 50 epoaches one layer with all the k means
fifty_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(50,i)
    fifty_epoches_one_layer.append(acc)

# get 50 epoaches three layer with all the k means
fifty_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(50,i)
    fifty_epoches_three_layer.append(acc)

# get 100 epoaches one layer with all the k means
hun_epoches_one_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_onelay(100,i)
    hun_epoches_one_layer.append(acc)

# get 100 epoaches three layer with all the k means
hun_epoches_three_layer = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_threelay(100,i)
    hun_epoches_three_layer.append(acc)

df = pd.DataFrame({
    'Number of clusters': [20, 50, 100, 200, 500, 1000, 2000],\
    '1 epochs one layer': one_epoches_one_layer,
    '1 epochs three layer': one_epoches_three_layer,
    '5 epochs one layer': five_epoches_one_layer,
    '5 epochs three layer': five_epoches_three_layer,
    '10 epochs one layer': ten_epoches_one_layer,
    '10 epochs three layer': ten_epoches_three_layer,
    '20 epochs one layer': twenty_epoches_one_layer,
    '20 epochs three layer': twenty_epoches_three_layer,
    '50 epochs one layer': fifty_epoches_one_layer,
    '50 epochs three layer': fifty_epoches_three_layer,
    '100 epochs one layer': hun_epoches_one_layer,
    '100 epochs three layer': hun_epoches_three_layer
})

df = df.melt('Number of clusters', var_name='cols',  value_name='Accuracy Rate')
g = sns.factorplot(x="Number of clusters", y="Accuracy Rate", hue='cols', data=df)

# get epoaches one layer with 500 k means
five_hun_one_layer = []
for i in [1, 5, 10, 20, 50, 100]:
    acc = get_acc_onelay(i, 500)
    five_hun_one_layer.append(acc)

# get epoaches one layer with 500 k means
five_hun_three_layer = []
for i in [1, 5, 10, 20, 50, 100]:
    acc = get_acc_threelay(i, 500)
    five_hun_three_layer.append(acc)

df2 = pd.DataFrame({
    'Epochs': [1, 5, 10, 20, 50, 100],
    'One layer': five_hun_one_layer,
    'Three layers': five_hun_three_layer})

df2 = df2.melt('Epochs', var_name='cols',  value_name='Accuracy Rate')
g = sns.factorplot(x="Epochs", y="Accuracy Rate", hue='cols', data=df2)


# Evaluate activate function
func = ['relu','sigmoid','tanh']
trial = [0,1,2,3,4]
accuracys = np.zeros((3,5))
for j, f in enumerate(func):
    for i in trial:
        y = pd.read_csv('data_folder/' + f + '_' + str(i)+'_trial_5epochs_threelay500.csv').values
        print(i)
        accuracys[j,i] = (y[:,0] ==y[:,1]).mean()
print(accuracys)


plt.plot(trial,accuracys[0,:],'-x',label='Relu')
plt.plot(trial,accuracys[1,:],'-o',label='Sigmoid')
plt.plot(trial,accuracys[2,:],'-X',label='Tanh')
plt.title('Accuracy of different Activate Function')
plt.xlabel('trial')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## Evaluate Logistic Regression
def get_acc_logistic(n_cluster):
    direct = 'data_folder/lyp' + str(n_cluster) + '.csv'
    true_label = pd.read_csv('data_folder/countfeature/test_feature20.csv')
    true = true_label.iloc[:, -1]
    df = pd.read_csv(direct)
    df['0.1'] = true
    matches = df[df['0']==df['0.1']]
    acc_rate = len(matches)/len(df)
    return acc_rate

LR = []
for i in [20, 50, 100, 200, 500, 1000, 2000]:
    acc = get_acc_logistic(i)
    LR.append(acc)

df3 = pd.DataFrame({
    'Number of clusters': [20, 50, 100, 200, 500, 1000, 2000],
    'Accuracy': LR,
})

df3 = df3.melt('Number of clusters', var_name='cols',  value_name='Accuracy')
g = sns.factorplot(x="Number of clusters", y="Accuracy", hue='cols', data=df3)


