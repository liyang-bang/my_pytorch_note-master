#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/28/2020 14:30'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""

'''
read json file
out put label file
separate label to train and val
***
only keep one pic in a period
'''
import datetime
import json
import shutil
import numpy as np


def read_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
        return data

# 计算数据各个内容的占比，返回字典
def show_proportion(json_file):
    dict = {}
    data = json.load(open(json_file))
    data = data["annotations"]
    for d in data:
        status = d["status"]
        if  status in dict.keys():
            dict[status] +=1
        else:
            dict[status] = 1
    print('show_proportion: ',dict)
    return dict


from imblearn.over_sampling import RandomOverSampler



def split_list(t_list, n):
    # list_1 = list()
    list_2 = list()
    if len(t_list) <= n:
        list_1 = list
    else:
        list_1 = t_list[0:n]
        list_2 = t_list[n:len(t_list)]
    return list_1, list_2


def generate_label_dic(path):
    data = json.load(open(path))
    key_label_set = set()
    data = data["annotations"]
    for dict in data:
        id = dict['id']
        key = dict["key_frame"]
        status = dict["status"]
        # print(id+'_'+key)
        s = str(status)
        key_label_set.add(id + '_' + s+'_'+key + ',' + str(status))
    return key_label_set

import os

def list_to_file(list, path):
    with open(path, 'w') as file_object:
        for line in list:
            file_object.write(str(line) + '\n')


def read_label_to_list(file):
    out_list = list()
    with open(file, 'r') as f:
        out_list.append(f.read())
    return out_list


def copy_img(name_list, origin_pics, path):
    # father label key
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

    for name in name_list:
        father_path = name.split('_')[0]
        pic_name = name.split('_')[2].split(',')[0]
        label = name.split('_')[1]
        origin_pic_path = origin_pics + '\\' + father_path + '\\' + pic_name
        new_pic_path = path + '\\'+father_path + '_' + label +'_'+ pic_name
        shutil.copy(origin_pic_path, new_pic_path)

def get_pics_path(father_path):
    g = os.walk(father_path)
    # path_list = set()
    path_file_list = set()
    for path, dir_list, file_list in g:
        for file_name in file_list:
            # print(os.path.join(path, file_name))
            if (file_name.find('DS_Store') == -1):
                path_file_list.add(os.path.join(path, file_name))
                # path_list.add(os.path)
    return path_file_list

def copy_file(root_dir,target_path):
    shutil.copy(root_dir,target_path)

def get_new_copy(dir,old_dir):

    if os.path.exists(dir):#os.path.join(dir)
        file_name = dir.split('.')[0].split('\\')[len(dir.split('.')[0].split('\\')) - 1]+ '.jpg'
        if len(dir.split('.')[0].split('\\')[len(dir.split('.')[0].split('\\')) - 1].split('_')) == 3:
            new_name = dir.split('.')[0].split('\\')[len(dir.split('.')[0].split('\\')) - 1] + '_'+'0' + '.jpg'
        else:
            num = int(dir.split('.')[0].split('\\')[len(dir.split('.')[0].split('\\')) - 1].split('_')[3]) +1
            new_name = dir.split('.')[0].split('\\')[len(dir.split('.')[0].split('\\')) - 1] +'_'+ str(num) + '.jpg'
        # print(dir.replace(file_name,new_name))
        get_new_copy(dir.replace(file_name,new_name),old_dir)
    else:
        print(dir,old_dir)
        shutil.copy(old_dir, dir)

from random import sample
def random_add_pics(path,num):
    list = get_pics_path(path)
    pick_list = sample(list, num)
    print(pick_list)

    for path in pick_list:
        get_new_copy(path,path)



def copy_img_by_type(name_list, origin_pics, path):
    # father label key
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

    for name in name_list:
        father_path = name.split('_')[0]
        pic_name = name.split('_')[2].split(',')[0]
        label = name.split('_')[1]
        origin_pic_path = origin_pics + '\\' + father_path + '\\' + pic_name

        if not os.path.exists(path + '\\' + label):
            os.mkdir(path + '\\' + label)

        # print(path + '\\'+label)
        new_pic_path = path + '\\'+label+'\\'+father_path + '_' + label +'_'+ pic_name
        shutil.copy(origin_pic_path, new_pic_path)


def main():
    json_file = 'D:\\Dataset\\amap_traffic_GaoDe\\amap_traffic_annotations_train.json'
    origin_pics = "D:\\Dataset\\amap_traffic_GaoDe\\0712_train_480-640"
    label_set = generate_label_dic(json_file)
    print(list(label_set)[0])  # 打印set中第一个元素看看，其实set中的元素是乱序的
    print('len(label_set)', len(label_set))


    # # save label.csv
    # # list1, list2 = split_list(list(label_set), 99999999)
    # filename_train = "D:\\Dataset\\amap_traffic_GaoDe\\all_train_480-640_over\\label.csv"
    # list_to_file(list(label_set),filename_train)
    # # filename_val = "D:\\Dataset\\amap_traffic_GaoDe\\val_480-640_more_over\\label.csv"
    # # list_to_file(list2, filename_val)
    #
    # # save pics
    # out_train = 'D:\\Dataset\\amap_traffic_GaoDe\\all_train_480-640_over\\pics'
    # copy_img_by_type(list(label_set), origin_pics, out_train)

    random_add_pics('D:\\Dataset\\amap_traffic_GaoDe\\all_train_480-640_over\\pics\\2\\',900)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print('done ', str(time_cost).split('.')[0])
