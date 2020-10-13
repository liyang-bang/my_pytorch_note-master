#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/28/2020 10:34'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""

'''
    Resize     
    RandomCrop
    CenterCrop 
    Normalize  
    
    image to tensor
    tensor to image
    
    save image to file
'''
import datetime
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


def get_img_size(img_path):
    # 先查看图片尺寸
    img = Image.open(img_path)
    img_size = img.size
    w = img.width
    h = img.height
    f = img.format
    img.close()
    # print(img_path,'size:',w,h,f)
    return img_size


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


def pics_size(pic_files):
    size_set = set()
    for f in pic_files:
        pic_size = get_img_size(f)
        size_set.add(pic_size)
    print('all pic kinds:', size_set)


def process_image(path, size=(480,640)):
    # print('resize:',size)
    mode = Image.open(path)
    # 使用Compose函数生成一个PiPeLine
    transform1 = transforms.Compose([
        transforms.Resize(size),
        # transforms.RandomCrop(224),
        # transforms.CenterCrop((size, size)),#中心截取
        transforms.ToTensor()  # 注意，只有PIL读取的图片才能被tranforms接受emmmm（有内鬼终止交易）
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    mode = transform1(mode)
    return mode


def save_tensor_image(path, mode):
    file_name = path.split('\\')[len(path.split('\\')) - 1]
    father_path = path.replace('\\' + file_name, '')
    # save pics to file
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    # tensor to image
    image = transforms.ToPILImage()(mode)
    image.save(path)


def showTorchImage(image):
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    plt.show()


def main():
    # get all pic path to set()
    path_list = get_pics_path('D:\\Dataset\\amap_traffic_GaoDe\\amap_traffic_train_0712')
    # print all kink of pics size
    pics_size(path_list)
    print(len(path_list))
    # separate to train and val

    # resize all pics
    for pic in path_list:
        # showTorchImage(read_image(pic))
        mode = process_image(pic)  # save to file
        new_pic = pic.replace('amap_traffic_train_0712', '0712_train_480-640')
        save_tensor_image(new_pic, mode)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
