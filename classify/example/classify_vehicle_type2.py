#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'ai_data_analysis'
__author__ = 'deagle'
__date__ = '2020/6/23 16:12'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓     ┏┓
            ┏━┛┻━━━━━┛┻━┓
            ┃     ☃     ┃
            ┃  ┳┛   ┗┳  ┃
            ┃     ┻     ┃
            ┗━┓       ┏━┛
              ┃       ┗━━━━┓
              ┃   神兽保佑  ┣┓
              ┃   永无BUG！ ┣┛
              ┗━┓┓┏━━━━┳┓┏━┛
                ┃┫┫    ┃┫┫
                ┗┻┛    ┗┻┛
"""


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn.datasets
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# In[3]:


import pandas as pd
# vehicle_file="D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_20200601_350500_1.8_0.2.csv"
vehicle_file="D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_3days_350500_1.2_0.65.csv"

vehicle_test_file = "D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_20200605_350500_test_1.2_0.65.csv"
data = pd.read_csv(vehicle_file, header=0)
test = pd.read_csv(vehicle_test_file, header=0)


# In[4]:


plt.scatter(data["gap1"], data["gap2"], s=40, c=data["type"], edgecolors ='g',cmap=plt.cm.binary)


# In[5]:


plt.scatter(test["gap1"], test["gap2"], s=40, c=test["type"], edgecolors ='r',cmap=plt.cm.binary)


# In[6]:


df1=pd.DataFrame(data)
df1 = df1.drop(['type'],axis=1)
X2 = df1.values
X2.shape


# In[7]:


# test
df2=pd.DataFrame(test)
df2 = df2.drop(['type'],axis=1)
X_t = df2.values
X_t.shape


# In[8]:


y1 = data["type"].tolist()
y2 = np.array(y1).T
y2.shape


# In[9]:


y_t0 = test["type"].tolist()
y_t = np.array(y_t0).T
y_t.shape


# In[10]:


X = torch.from_numpy(X2).type(torch.FloatTensor)
y = torch.from_numpy(y2).type(torch.LongTensor)

X_test = torch.from_numpy(X_t).type(torch.FloatTensor)
y_test = torch.from_numpy(y_t).type(torch.LongTensor)


# ### 使用激活函数 tanh

# In[11]:


from IPython.display import Latex
Latex(r"$tanhx = \frac{\sin hx}{\cos hx} = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$")


# In[14]:


# our class must extend nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        # This applies Linear transformation to input data.
        self.fc1 = nn.Linear(2, 3)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(3, 2)

    # This must be implemented
    def forward(self, x):
        # Output of the first layer
        x = self.fc1(x)
        # Activation function is Relu. Feel free to experiment with this
        x = torch.tanh(x)
        # This produces output
        x = self.fc2(x)
        return x

    # This function takes an input and predicts the class, (0 or 1)
    def predict(self, x):
        # Apply softmax to output
        pred = F.softmax(self.forward(x),dim=1)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


# In[24]:


# Initialize the model
model = Net()
# Define loss criterion
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Number of epochs
epochs = 500
# List to store losses
losses = []
for i in range(epochs):
    # Precit the output for Given input
    y_pred = model.forward(X)
    # Compute Cross entropy loss
    loss = criterion(y_pred, y)
    # Add loss to the list
    losses.append(loss.item())
    # Clear the previous gradients
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Adjust weights
    optimizer.step()

from sklearn.metrics import accuracy_score

print(accuracy_score(model.predict(X), y))


# In[25]:


print(accuracy_score(model.predict(X_test), y_test))


# In[16]:


def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y,edgecolors ='b', cmap=plt.cm.binary)


# In[20]:


plot_decision_boundary(lambda x: predict(x), X.numpy(), y.numpy())


# In[17]:


plot_decision_boundary(lambda x: predict(x), X_t.numpy(), y_t.numpy())

