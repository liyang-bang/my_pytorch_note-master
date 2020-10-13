#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/29/2020 17:10'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
from torch.utils import data
import numpy as np
import torch
import cv2
import pandas as pd
import torch.nn as nn
import os
import json

# 5. 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

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

# 4. 我们通过继承Dataset类来创建我们自己的"数据加载类"，命名为FaceDataset。
class amap_dataset(data.Dataset):

    def __init__(self, root):
        super(amap_dataset, self).__init__()
        self.path = root

    def folder(self):
        return self.path

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        pic = cv2.imread(self.path)
        res = cv2.resize(pic, dsize=(64, 48), interpolation=cv2.INTER_CUBIC)
        pic_normalized = res.reshape(3, 64, 48) / 255.0  # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        pic_tensor = torch.from_numpy(pic_normalized)  # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        pic_tensor = pic_tensor.type('torch.FloatTensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        return pic_tensor

    def __len__(self):
        return len(str(self.path))


def get_new_json(filepath, key, value):
    key_ = key.split(".")
    key_length = len(key_)
    with open(filepath, 'rb') as f:
        json_data = json.load(f)
        i = 0
        a = json_data
        while i < key_length:
            if i + 1 == key_length:
                a[key_[i]] = value
                i = i + 1
            else:
                a = a[key_[i]]
                i = i + 1
    f.close()
    return json_data


def rewrite_json_file(filepath, json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f)
    f.close()




# 验证模型在验证集上的正确率
def preidct(model, dataset, batch_size,json_file):

    val_loader = data.DataLoader(dataset, batch_size)
    folder = dataset.folder()
    # print(folder.type)
    fold = folder.split('\\')[len(folder.split('\\'))-1].split('_')[0]
    # f = folder.split('\\')[len(folder.split('\\'))-1].split('_')[1]

    # print(fold)


    for images in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        # print(pred[0])
        f = open(json_file)
        json_data = json.load(f)
        a = json_data
        dict2 = a["annotations"]
        for d in dict2:
            # id = d['id']
            if d['id'] == fold:
                d["status"] = str(pred[0])
        f.close()

        with open(json_file, 'w') as f2:
            json.dump(a, f2)
        f2.close()

        # with open(test_json, 'rb') as f:
        #     json_data = json.load(f)
        #     for item in json_data:
        #         item2 = item["annotations"]
        #                 for ite in
        #             item["ver"] = app_version
        #         out = file(json_file, 'w+')
        #         reContent = json.dump(json_data, out, ensure_ascii=False, indent=4)

        return

# get all pic absolute path
def get_pics_path(father_path):
    g = os.walk(father_path)
    path_file_list = set()
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if (file_name.find('DS_Store') == -1):
                path_file_list.add(os.path.join(path, file_name))
    return path_file_list

from shutil import copyfile

def main():
    pic_path = 'D:\\Dataset\\amap_traffic_GaoDe\\test_pics'
    json = 'D:\\Dataset\\amap_traffic_GaoDe\\amap_traffic_annotations_test.json'
    # 修改输出
    out_json = 'D:\\Dataset\\amap_traffic_GaoDe\\predict\\net10_predict.json'

    copyfile(json, out_json)

    model2 = amap_cnn()
    model2.load_state_dict(torch.load(r'D:\Dataset\amap_traffic_GaoDe\cnn_model\checkpoint\23-05-27\checkpoint.pt'))
    model2.eval()

    path_file_list = get_pics_path(pic_path)
    print('pic quantity: ', len(path_file_list))

    # res_list = list()
    for p in path_file_list:
        data_set = amap_dataset(root=p)
        preidct(model2,data_set,1,out_json)


        # print('2')
        # pic = cv2.imread(p)
        # res = cv2.resize(pic, dsize=(64, 48), interpolation=cv2.INTER_CUBIC)
        # pic_normalized = res.reshape(3, 64, 48) / 255.0  # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # pic_tensor = torch.from_numpy(pic_normalized)  # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        # pic_tensor = pic_tensor.type('torch.FloatTensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        # print(pic_tensor.shape)
        # model2.train()
        # pred = model2.forward(pic_tensor)
        # res = print(pred)

    # dataset = amap_dataset(root=pic_path) #tensor
    # preidct(model2, dataset, batch_size=16,test_json,result_json)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
