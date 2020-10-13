#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/28/2020 15:59'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""

'''
整理图片，按照Label
'''
import datetime
import shutil


def read_label_to_list(file):
    out_list = list()
    with open(file, 'r') as f:
        out_list.append(f.read())
    return out_list


def copy_img(name_list,origin_pics, path):
    for name in name_list:
        father_path = name.split(',')[0]
        pic_name = name.split(',')[1]
        label = name.split(',')[2]
        origin_pic_path = origin_pics+'\\'+father_path+'\\'+pic_name
        new_pic_path = path+'\\'+father_path+'_'+pic_name
        shutil.copy(origin_pic_path,new_pic_path)


def main():
    train_file = 'D:\\Dataset\\amap_traffic_GaoDe\\train\\label.csv'
    val_file = 'D:\\Dataset\\amap_traffic_GaoDe\\val\\label.csv'

    out_train = 'D:\\Dataset\\amap_traffic_GaoDe\\train\\pics'
    out_val = 'D:\\Dataset\\amap_traffic_GaoDe\\val\\pics'

    origin_pics = "D:\\Dataset\\amap_traffic_GaoDe\\0712_train"

    train_list = read_label_to_list(train_file)
    val_list = read_label_to_list(val_file)

    copy_img(train_list, origin_pics,out_train)
    copy_img(val_list, origin_pics,out_val)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
