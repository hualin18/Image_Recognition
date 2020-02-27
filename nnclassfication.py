from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data
import csv


def one_layer_nn(tr_path,te_path,inputdim,num_epochs=5):
    df_tr = pd.read_csv(tr_path)
    df_te = pd.read_csv(te_path)
    features_tr = df_tr[df_tr.columns[0:-1]].values
    y_tr = df_tr[df_tr.columns[-1]].values
    features_te = df_te[df_te.columns[0:-1]].values
    y_te = df_te[df_te.columns[-1]].values
    tr_tensor = torch.Tensor(features_tr)
    te_tensor = torch.Tensor(features_te)
    tr_dataset = [tuple((tr_tensor[i],y)) for i,y in enumerate(y_tr)]
    te_dataset = [tuple((te_tensor[i],y)) for i,y in enumerate(y_te)]

    batch_size = 100
    tr_loader = torch.utils.data.DataLoader(dataset = tr_dataset, batch_size = batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(dataset = te_dataset, batch_size = batch_size, shuffle=False)

    # Define one layer model
    outputdim=20
    learning_rate = 0.001
    class Logistic_TorchModel(nn.Module):
        def __init__(self,inputdim,outputdim):
            super(Logistic_TorchModel,self).__init__()
            self.linear = nn.Linear(inputdim, outputdim)
        def forward(self, x):
            out = self.linear(x)
            return out

    model = Logistic_TorchModel(inputdim, outputdim)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    # fit
    total_step=len(tr_loader)
    for epoch in range(num_epochs):
        for i, (feature, labels) in enumerate(tr_loader):
            feature = feature.requires_grad_()
            labels = labels
            # Initial grad
            optimizer.zero_grad()
            # forward
            outputs = model(feature)
            # judge
            loss = criteria(outputs, labels)
            loss.backward()
            # update W
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    y_pred = []
    la = []
    with torch.no_grad():
        c=0
        t=0

        for i, (feature, labels) in enumerate(te_loader):
            images = feature.requires_grad_()
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            c += (pred==labels).sum()
            t += labels.size(0)
            y_pred.append(pred.numpy())
            la.append(labels.numpy())
        print(float(c)/float(t) *100)
        pd.concat([pd.DataFrame(np.hstack(y_pred)),pd.DataFrame(np.hstack(la))],axis=1).to_csv('data_folder/' + str(num_epochs) + 'epochs_onelay'+str(inputdim)+'.csv',index=False)
## Three layers Feedforward Neural Network
def three_layer_nn(tr_path,te_path,inputdim, num_epochs=5,Func=nn.ReLU(),store_path='data_folder/'):
    df_tr = pd.read_csv(tr_path)
    df_te = pd.read_csv(te_path)
    features_tr = df_tr[df_tr.columns[0:-1]].values
    y_tr = df_tr[df_tr.columns[-1]].values
    features_te = df_te[df_te.columns[0:-1]].values
    y_te = df_te[df_te.columns[-1]].values
    tr_tensor = torch.Tensor(features_tr)
    te_tensor = torch.Tensor(features_te)
    tr_dataset = [tuple((tr_tensor[i],y)) for i,y in enumerate(y_tr)]
    te_dataset = [tuple((te_tensor[i],y)) for i,y in enumerate(y_te)]

    batch_size = 50
    tr_loader = torch.utils.data.DataLoader(dataset = tr_dataset, batch_size = batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(dataset = te_dataset, batch_size = batch_size, shuffle=False)

    # Define three layer model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 100
    outputdim=20
    learning_rate = 0.001
    hidden_size = int((inputdim+outputdim)/2)

    class Three_NeuralNet(nn.Module):
        def __init__(self, inputdim, hidden_size, outputdim):
            super(Three_NeuralNet, self).__init__()
            self.fc1 = nn.Linear(inputdim, hidden_size)
            self.func = Func
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.func = Func
            self.fc3 = nn.Linear(hidden_size, outputdim)

        def forward(self, x):
            out = self.fc1(x)
            out = self.func(out)
            out = self.fc2(out)
            out = self.func(out)
            out = self.fc3(out)
            return out

    model = Three_NeuralNet(inputdim, hidden_size, outputdim).to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # fit
    total_step = len(tr_loader)
    for epoch in range(num_epochs):
        for i, (feature, labels) in enumerate(tr_loader):
            # Move tensors to the configured device
            images = feature.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criteria(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    y_pred = []
    la = []
    with torch.no_grad():
        c=0
        t=0
        for i, (images, labels) in enumerate(te_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            y_pred.append(pred.cpu().numpy())
            la.append(labels.cpu().numpy())
            c += (pred == labels).sum()
            t += labels.size(0)

        print(float(c) / float(t) * 100)
    pd.concat([pd.DataFrame(np.hstack(y_pred)), pd.DataFrame(np.hstack(la))],axis=1).to_csv(
        store_path + str(num_epochs) + 'epochs_threelay' + str(inputdim) + '.csv', index=False)


