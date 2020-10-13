#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/29/2020 10:11'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import pandas as pd
import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils import data
from torch.autograd import Variable
from sklearn.metrics import f1_score
import torch.optim as optim




class amap_loss(nn.Module):
    def __init__(self):
        super(amap_loss, self).__init__()

    def forward(self, predict, labels):
        predict_np = np.argmax(predict.data.cpu().numpy(), axis=1)
        label_np = labels.data.numpy()
        f1 = f1_score(label_np, predict_np, average=None)
        # print(f1)

        list1 = label_np.tolist()
        list2 = predict_np.tolist()
        list_all = list1 + list2
        num_set = set(list_all)
        # print(num_set)

        amap_f1 = 0.0
        if num_set == {0, 1, 2}:
            amap_f1 = 1 - (f1[0] * 0.2 + f1[1] * 0.2 + f1[2] * 0.6)
        elif num_set == {0, 1}:
            amap_f1 = 1 - (f1[0] * 0.2 + f1[1] * 0.2)
        elif num_set == {0, 2} or num_set == {1, 2}:
            amap_f1 = 1 - (f1[0] * 0.2 + f1[1] * 0.6)
        elif num_set == {0} or num_set == {1}:
            print('0/1')
            amap_f1 = 1 - (f1[0] * 0.2)
        elif num_set == {2}:
            print('2')
            amap_f1 = 1 - (f1[0] * 0.6)

        arr_f1 = np.array([amap_f1])
        f1_tensor = torch.autograd.Variable(torch.from_numpy(arr_f1), requires_grad=True)
        # print(f1_tensor)
        return f1_tensor



# 4. 我们通过继承Dataset类来创建我们自己的"数据加载类"，命名为FaceDataset。
class amap_dataset(data.Dataset):
    # 首先要做的是类的初始化。之前的data-label对照表已经创建完毕，在加载数据时需用到其中的信息。因此在初始化过程中，我们需要完成对data-label对照表中数据的读取工作。
    # 通过pandas库读取数据，随后将读取到的数据放入list或numpy中，方便后期索引。
    # 初始化
    def __init__(self, root):
        super(amap_dataset, self).__init__()
        # root为train或val文件夹的地址
        self.root = root
        # 读取data-label对照表中的内容
        df_path = pd.read_csv(root + '/label.csv', header=None, usecols=[0])  # 读取第一列文件名
        df_label = pd.read_csv(root + '/label.csv', header=None, usecols=[1])  # 读取第二列label
        # 将其中内容放入numpy，方便后期索引
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        pic = cv2.imread(self.root + '/pics/' + self.path[item])  # cv2.imread(fp)读取图片,得到的是BGR颜色空间的numpy类型，uint8类型
        # print(pic.shape)
        # 调整大小
        # pic_res = cv2.resize(pic, (64,48))
        # print(pic_res.shape)
        res = cv2.resize(pic, dsize=(64, 48), interpolation=cv2.INTER_CUBIC)
        # 图像剪裁
        # res_c = res[:, 4:60]
        # print(res.shape)
        # print(res.shape)
        # 读取单通道灰度图
        # face_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # print(face_gray.shape)
        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # 直方图均衡化
        # pic_hist = cv2.equalizeHist(res)
        # 像素值标准化
        pic_normalized = res.reshape(3, 64, 48) / 255.0  # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # print(pic_normalized.shape)
        # 用于训练的数据需为tensor类型
        pic_tensor = torch.from_numpy(pic_normalized).cuda()  # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        pic_tensor = pic_tensor.type('torch.FloatTensor').cuda()  # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        # print('pic_tensor.type()',pic_tensor.type())
        label = self.label[item]
        # label_cuda = label.cuda()
        return pic_tensor, label

    def __len__(self):
        return self.path.shape[0]


# 5. 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 6. 模型编写
class amap_cnn(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(amap_cnn, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            # input:(bitch_size, 1, 64, 48), output:(bitch_size, 64, 64, 48), (64-3+2*1)/1+1 = 64
            # input:(bitch_size, 1, 256, 144), output:(bitch_size, 64, 256, 144), (64-3+2*1)/1+1 = 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            # output(bitch_size, 64, 128, 72)
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 32, 24), output:(bitch_size, 128, 32, 24), (32-3+2*1)/1+1 = 32
            # input:(bitch_size, 64, 128, 72), output:(bitch_size, 128, 128, 72), (32-3+2*1)/1+1 = 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),  # 归一化
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 64 ,36)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 64, 36), output:(bitch_size, 256, 64, 36), (16-3+2*1)/1+1 = 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),  # 归一化
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 32 ,18)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # # 第4次卷积、池化
        # self.conv4 = nn.Sequential(
        #     # input:(bitch_size, 128, 64, 36), output:(bitch_size, 256, 64, 36), (16-3+2*1)/1+1 = 64
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(num_features=256),# 归一化
        #     nn.RReLU(inplace=True),
        #     # output:(bitch_size, 256, 32 ,18)
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        # self.conv4.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=256 * 8 * 6, out_features=4096),
            # nn.RReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Linear(in_features=4096, out_features=1024),
            # nn.RReLU(inplace=True),
            # nn.Linear(in_features=1024, out_features=256),
            # nn.RReLU(inplace=True),
            nn.Linear(in_features=256 * 8 * 6, out_features=3),
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


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    # print(len(val_loader))
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = amap_cnn().cuda()
    # 损失函数
    # loss_function = nn.CrossEntropyLoss().cuda()
    loss_function = amap_loss().cuda()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step() # 学习率衰减
        model.train()  # 模型训练
        for images, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images).cpu()
            # 误差计算
            # loss_rate = loss_function(output, labels).cpu()
            loss_rate = loss_function(output, labels).cpu()
            # 误差的反向传播
            loss_rate.backward()
            # print(loss_rate.backward())
            # 更新参数
            optimizer.step()

        import time
        time_format = time.strftime('%H-%M-%S', time.localtime(time.time()))
        # 打印每轮的损失
        # print('After {} epochs , loss_rate: '.format(epoch + 1), loss_rate.item())
        if epoch % 5 == 0 and epoch > 0:
            model.eval()  # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            # print('After {} epochs , acc_train: '.format(epoch + 1), acc_train)
            # print('After {} epochs , acc_val: '.format(epoch + 1), acc_val)
            print('Epoch: ', epoch, '| loss_rate: %.8f' % loss_rate,
                  '| acc_train: %.4f' % acc_train,
                  '| val accuracy: %.4f' % acc_val, '|', time_format)
        else:
            print('Epoch: ', epoch, '| loss_rate: %.8f' % loss_rate)

        if loss_rate < 0.01:
            return model
        import os
        if epoch % 5 == 0 and epoch > 0:
            # 中途保存模型，这样就可以随时启停喽~
            new_path = "D:\\Dataset\\amap_traffic_GaoDe\\cnn_model\\checkpoint\\" + time_format
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            torch.save(model.state_dict(), new_path+"\\checkpoint.pt")
            print('model saved',time_format)

    return model


def main():
    train_path = 'D:/Dataset/amap_traffic_GaoDe/train_144-256_more_5600'
    val_path = 'D:/Dataset/amap_traffic_GaoDe/val_144-256'
    # generate_label(train_path)
    # generate_label(val_path)

    # 数据集实例化(创建数据集)
    train_dataset = amap_dataset(root=train_path)
    val_dataset = amap_dataset(root=val_path)
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=100, epochs=600, learning_rate=0.1, wt_decay=0)
    # 保存模型
    # torch.save(model, 'D:/Dataset/amap_traffic_GaoDe/cnn_model/model_net5.pkl')
    torch.save(model.state_dict(), 'D:/Dataset/amap_traffic_GaoDe/cnn_model/model_net9.pt')


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
