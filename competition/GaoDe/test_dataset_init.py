#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/29/2020 17:52'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import json
import shutil


def read_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())
        return data


def split_list(t_list, n):
    list_1 = list()
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
        # label = id["key_frame"]
        key_label_set.add(id + '_' + key + ',' + str(status))
    return key_label_set

def list_to_file(list,path):
    with open(path, 'w') as file_object:
        for line in list:
            file_object.write(str(line)+'\n')



def read_label_to_list(file):
    out_list = list()
    with open(file, 'r') as f:
        out_list.append(f.read())
    return out_list


def copy_img(name_list,origin_pics, path):
    for name in name_list:
        father_path = name.split('_')[0]
        pic_name = name.split('_')[1].split(',')[0]
        label = name.split('_')[1].split(',')[1]
        origin_pic_path = origin_pics+'\\'+father_path+'\\'+pic_name
        new_pic_path = path+'\\'+father_path+'_'+pic_name
        shutil.copy(origin_pic_path,new_pic_path)




def main():
    json_file = 'D:\\Dataset\\amap_traffic_GaoDe\\amap_traffic_annotations_test.json'
    label_set = generate_label_dic(json_file)
    print(list(label_set)[0])  # 打印set中第一个元素看看，其实set中的元素是乱序的
    print('len(label_set)', len(label_set))

    out_pic_index = "D:\\Dataset\\amap_traffic_GaoDe\\test_data_prepare\\pic_index.csv"
    list_to_file(list(label_set), out_pic_index)

    origin_pics = "D:\\Dataset\\amap_traffic_GaoDe\\amap_traffic_test_0712"
    out_test_pic = 'D:\\Dataset\\amap_traffic_GaoDe\\test_pics'
    copy_img(list(label_set), origin_pics, out_test_pic)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])