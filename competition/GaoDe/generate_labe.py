#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '8/3/2020 22:36'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import os

def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        #print(com_path)
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    # print(path_list)
    return path_list

def generate_label_file(path_list,out):
    paths = set()
    for p in path_list:
        file_name = str(p).split('\\')[len(str(p).split('\\'))-1]
        label = str(p).split('\\')[len(str(p).split('\\'))-1].split('_')[1]
        paths.add(file_name+','+label)

    with open(out, 'w') as file_object:
        for line in paths:
            file_object.write(str(line)+'\n')

def main():
    paths = get_all_path(r'D:\Dataset\amap_traffic_GaoDe\over\train\pics')
    generate_label_file(paths,r'D:\Dataset\amap_traffic_GaoDe\over\train\label.csv')


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
