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
import sklearn.datasets
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pandas as pd
# vehicle_file="D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_20200601_350500_1.8_0.2.csv"
vehicle_file="D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_3days_350500_1.2_0.65.csv"

vehicle_test_file = "D:/work_source/test_env/BadCaseAnalysis/carTypeOutput/vehicle_ID_20200605_350500_test_1.2_0.65.csv"
data = pd.read_csv(vehicle_file, header=0)
test = pd.read_csv(vehicle_test_file, header=0)

plt.scatter(data["gap1"], data["gap2"], s=40, c=data["type"], edgecolors ='g',cmap=plt.cm.binary)
p1 = plt.show()

plt.scatter(test["gap1"], test["gap2"], s=40, c=test["type"], edgecolors ='r',cmap=plt.cm.binary)
p2 = plt.show()



if __name__ == '__main__':
    print('vision')