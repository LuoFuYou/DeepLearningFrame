from graph import PyGraph
from variable import Variable
from linear import PyLinear
from act_func import PyReLU
from loss_func import PyMSELoss, PyCELoss
import random
import numpy as np
import os
import sys
import time
import visdom
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# vis = visdom.Visdom(env = 'plot1')

dataset = load_iris()
data = dataset.data
target = dataset.target

standard_scalar = StandardScaler()
data = standard_scalar.fit_transform(data)

lb = LabelBinarizer()
target = lb.fit_transform(target)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size= 0.3, random_state= 42)

graph = PyGraph()

X0 = Variable(graph)
Y = Variable(graph)
graph << X0
graph << Y

lin1 = PyLinear(graph, 4, 10)
relu1 = PyReLU(graph)

lin2 = PyLinear(graph, 10, 3)
relu2 = PyReLU(graph)

loss = PyCELoss(graph)

X1 = lin1(X0)
X2 = relu1(X1)
X3 = lin2(X2)
P = relu2(X3)
L = loss(P, Y)

itr = 1
lr = 0.01
itr_list = []
loss_list = []
while itr < 10:
    all_loss = []
    for x, y in zip(X_train, Y_train):
        X0.SetData(x[np.newaxis, :, np.newaxis])
        Y.SetData(np.expand_dims(y, 0))
        
        graph.ZeroGrad()
        graph.Forward()
        graph.Backward()
        graph.UpdateParams(lr)
        
        all_loss.append(L.data_node.data.item())
    
    itr_list.append(itr)
    loss_list.append(np.mean(all_loss))
    itr += 1
    
print(loss_list)
    # plt.plot(itr_list, loss_list)
    # plt.show()
    # time.sleep(0.5)
    # plt.close()