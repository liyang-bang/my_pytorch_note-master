#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/27/2020 11:08'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
# from __future__ import print_function

import os
import pandas as pd
import cv2
import numpy as np


# https://www.cnblogs.com/HL-space/p/10888556.html

# 1. 读入csv数据（图像的像素点数据）
def init():
    # 将label和像素数据分离
    print('step 1. read dataset file to data frame.')
    # 修改为train.csv在本地的相对或绝对地址
    path = 'D:/Dataset/FaceData/train.csv'
    # 读取数据
    df = pd.read_csv(path)
    # 提取label数据
    df_y = df[['label']]
    # 提取feature（即像素）数据
    df_x = df[['feature']]
    # 将label写入label.csv
    df_y.to_csv('D:/Dataset/FaceData/label.csv', index=False, header=False)
    # 将feature数据写入data.csv
    df_x.to_csv('D:/Dataset/FaceData/data.csv', index=False, header=False)


# 2. 数据可视化
def data_to_pic():
    print('step 2. convert data to pic and save to path')
    # 指定存放图片的路径
    path = 'D:/Dataset/FaceData/pics'
    # 读取像素数据
    data = np.loadtxt('D:/Dataset/FaceData/data.csv')

    # 按行取数据
    for i in range(data.shape[0]):
        face_array = data[i, :].reshape((48, 48))  # reshape
        cv2.imwrite(path + '/' + '{}.jpg'.format(i), face_array)  # 写图片

    # 输出两份数据，一个是train,另一个是val
    for i in range(24000):
        face_array = data[i, :].reshape((48, 48))  # reshape
        cv2.imwrite('D:/Dataset/FaceData/train/pics/' + '{}.jpg'.format(i), face_array)  # 写图片
        # 输出数据是三通道，（每个通道都完全一样，都是灰色）
        # 后面根据 cv2.COLOR_BGR2GARY 可以仅读取灰色通道数据

    for i in range(24000, data.shape[0]):
        face_array = data[i, :].reshape((48, 48))  # reshape
        cv2.imwrite('D:/Dataset/FaceData/val/pics/' + '{}.jpg'.format(i), face_array)  # 写图片


# 3. 在 pytorch 创建数据集
def generate_label(path):
    print('step 3. load data, separate data to train and val')
    pis_path = path + '/pics'
    # 读取label文件
    df_label = pd.read_csv('D:/Dataset/FaceData/label.csv', header=None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(pis_path)
    # 用于存放图片名
    path_list = []
    # 用于存放图片对应的label
    label_list = []
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的label取出，分别放入path_list和label_list这两个列表中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            label_list.append(df_label.iat[index, 0])

    # 将两个列表写进dataset.csv文件
    path_s = pd.Series(path_list, dtype=str)
    label_s = pd.Series(label_list, dtype=str)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '/label.csv', index=False, header=False)


'''
pytorch 的 Dataset 类
Dataset类是Pytorch中图像数据集中最为重要的一个类，也是Pytorch中所有数据集加载类中应该继承的父类。
其中父类中的两个私有成员函数getitem()和len()必须被重载，否则将会触发错误提示。
其中getitem()可以通过索引获取数据，len()可以获取数据集的大小。
'''

# 4. 我们通过继承Dataset类来创建我们自己的"数据加载类"，命名为FaceDataset。
import torch
from torch.utils import data


class FaceDataset(data.Dataset):
    # 首先要做的是类的初始化。之前的data-label对照表已经创建完毕，在加载数据时需用到其中的信息。因此在初始化过程中，我们需要完成对data-label对照表中数据的读取工作。
    # 通过pandas库读取数据，随后将读取到的数据放入list或numpy中，方便后期索引。
    # 初始化
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        # root为train或val文件夹的地址
        self.root = root
        # 读取data-label对照表中的内容
        df_path = pd.read_csv(root + '/label.csv', header=None, usecols=[0])  # 读取第一列文件名
        df_label = pd.read_csv(root + '/label.csv', header=None, usecols=[1])  # 读取第二列label
        # 将其中内容放入numpy，方便后期索引
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    '''
    # 由于是读取图片数据，因此仍然借助opencv库。需要注意的是，之前可视化数据部分将像素值恢复为人脸图片并保存，
    # 得到的是3通道的灰色图（每个通道都完全一样），而在这里我们只需要用到单通道，因此在图片读取过程中，
    # 即使原图本来就是灰色的，但我们还是要加入参数从cv2.COLOR_BGR2GARY，保证读出来的数据是单通道的。
    # 读取出来之后，可以考虑进行一些基本的图像处理操作，如通过高斯模糊降噪、
    # 通过直方图均衡化来增强图像等（经试验证明，在本次作业中，直方图均衡化并没有什么卵用，
    # 而高斯降噪甚至会降低正确率，可能是因为图片分辨率本来就较低，模糊后基本上什么都看不清了吧）。
    # 读出的数据是48X48的，而后续卷积神经网络中nn.Conv2d() API所接受的数据格式是(batch_size, channel, width, higth)，
    # 本次图片通道为1，因此我们要将48X48 reshape为1X48X48。
    '''

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        face = cv2.imread(self.root + '/pics/' + self.path[item])
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)
        # 像素值标准化
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0  # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # 用于训练的数据需为tensor类型
        face_tensor = torch.from_numpy(face_normalized)  # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.FloatTensor')  # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        label = self.label[item]
        return face_tensor, label

    def __len__(self):
        return self.path.shape[0]


# 5. 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 6. 模型编写
import torch.nn as nn


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            # output(bitch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y



# 人脸旋转，尝试过但效果并不好，本次并未用到
def imgProcess(img):
    # 通道分离
    (b, g, r) = cv2.split(img)
    # 直方图均衡化
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    # 顺时针旋转15度矩阵
    M0 = cv2.getRotationMatrix2D((24,24),15,1)
    # 逆时针旋转15度矩阵
    M1 = cv2.getRotationMatrix2D((24,24),15,1)
    # 旋转
    gH = cv2.warpAffine(gH, M0, (48, 48))
    rH = cv2.warpAffine(rH, M1, (48, 48))
    # 通道合并
    img_processed = cv2.merge((bH, gH, rH))
    return img_processed

# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


import torch.optim as optim
def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step() # 学习率衰减
        model.train() # 模型训练
        for images, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        if epoch % 5 == 0:
            model.eval() # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)

    return model



if __name__ == '__main__':
    # init()
    # data_to_pic()
    train_path = 'D:/Dataset/FaceData/train'
    val_path = 'D:/Dataset/FaceData/val'
    # generate_label(train_path)
    # generate_label(val_path)

    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset(root='D:/Dataset/FaceData/train')
    val_dataset = FaceDataset(root='D:/Dataset/FaceData/val')
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=128, epochs=5, learning_rate=0.1, wt_decay=0)
    # 保存模型
    torch.save(model, 'model_net1.pkl')

