#!/usr/bin/env python
# coding: utf-8
# 2020-9-14 11:48:53
import matplotlib.pyplot as plt
# from scipy import stats
import numpy as np
import pandas as pd
import sys
from scipy import stats
import datetime

import os


def getLinkID(a, b):
    if str(b).startswith('-'):
        return str(a) + "1" + str(b)[1:]
    return str(a) + '0' + str(b)


# 计算小车均值和大车均值比例系数

def peak_peak(arr):
    try:
        # print(type(arr))
        # print(arr)
        if (len(arr) > 1):
            # print('=============================================================')
            # print(arr[0])
            # print(arr[1])
            return round(arr[0] / arr[1], 2)
        else:
            return 0.00
    except Exception as e:
        print(repr(e))
        # print(arr)
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def getCar_ff_truck_ff_key(a, b):
    return str(a) + '_' + str(b)


# ### 拼接自由流类别方法
def getCar_truck_key(a, b):
    return str(a) + '_' + str(b)


def process(dfAll, speedFilter, baseFile):
    outJpgPath = baseFile + 'jpg-' + str(speedFilter) + '/'
    if not os.path.exists(outJpgPath):
        os.makedirs(outJpgPath)
    saveRoidPathF = baseFile + 'rate/'
    print(outJpgPath)
    print(saveRoidPathF)
    if not os.path.exists(saveRoidPathF):
        os.makedirs(saveRoidPathF)
    saveRoidPath = saveRoidPathF + str(speedFilter) + '_rateUpSpeed.csv'
    # print(dfAll.head())
    # 按照linkID、大车自由流类别、小车自由流类别、时间、车辆类型组，并计算均值
    dfGroup = dfAll.groupby(['linkID', 'car_ff', 'truck_ff', 'min_key', 'vehicle_type'])['speed_sample'].apply(
        lambda x: np.mean(x.tolist()))
    print(dfGroup.head())

    # 结果转换为DF
    # dfGroup1 = dfGroup.reset_index()  # 如何将groupby之后的groupby对象转化为dataframe
    dfGroup1 = pd.DataFrame(dfGroup)
    print(dfGroup1.head())
    # ### 分组后求大小车提速比例
    dfGroup2 = dfGroup1.groupby(['linkID', 'car_ff', 'truck_ff', 'min_key']).agg(peak_peak)
    dfGroup3 = dfGroup2.reset_index()  # 如何将groupby之后的groupby对象转化为dataframe

    # dfGroup3.to_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/'+city+'/resaaa.csv',index=False,header=None)
    # 过滤NAN数据
    print('dfGroup3')
    print(dfGroup3.head())

    dfGroup6 = dfGroup3[np.isnan(dfGroup3['speed_sample']) == False]
    print('dfGroup3-1')
    print(dfGroup3.head())
    if len(dfGroup3) == 0:
        return;


    dfGroup6 = dfGroup6[dfGroup6['speed_sample'] != 0]
    print('dfGroup6')
    # print(dfGroup6.info())
    print(dfGroup6.head(100))
    if len(dfGroup6) == 0:
        return;

    dfGroup6['car_ff_truck_ff_key'] = dfGroup6.apply(lambda row: getCar_ff_truck_ff_key(row['car_ff'], row['truck_ff']),
                                                     axis=1).astype('str')

    # 获取所有的自由流分类

    dfGroupCar_ff_truck_ff_key = dfGroup6['car_ff_truck_ff_key'].drop_duplicates()
    # dfGroupCar_ff_truck_ff_key
    len(dfGroupCar_ff_truck_ff_key)

    # ### 保存每个类别自由流大小车提速比例和核函数图

    with open(saveRoidPath, 'w', encoding='utf-8') as fw:  # .users是一个临时文件
        for key in dfGroupCar_ff_truck_ff_key:
            try:
                df9_9 = dfGroup6[dfGroup6['car_ff_truck_ff_key'] == key]

                rate = stats.mode(df9_9['speed_sample'].tolist())[0][0]
                print(rate)
                fw.write(str(rate) + ',' + str(key)+','+len(df9_9['speed_sample'].tolist()))
                print(str(rate) + ',' + str(key)+','+len(df9_9['speed_sample'].tolist()))
                fw.write("\n")
                plt.figure(figsize=(16, 6))
                df9_9['speed_sample'].plot.hist(bins=250, alpha=0.5)
                df9_9['speed_sample'].plot(kind='kde', secondary_y=True)
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
                plt.rcParams['axes.unicode_minus'] = False
                plt.title('自由流类型: ' + str(key) + ' ' + str(rate))  # 直方图名称
                # plt.xlim(-0.5,5)
                plt.savefig(outJpgPath + str(key) + '.jpg')
                plt.show()
            except:
                print(str(key) + ' ' + str(rate))

    print('END')


def main():
    # ### 配置读取文件路径
    # 1.读取大车自由流=小车自由流link
    # 2.读取同根link同分钟不同自由流类别车辆速度数据
    # 3.拼接全linkID
    # city = sys.argv[1]
    #
    # baseFile = sys.argv[1] + '/'
    # readFile = sys.argv[2]
    # readFile1 = sys.argv[3]
    # roadClass = sys.argv[4]

    baseFile = r'./'
    readFile = r'./ssssssss.csv'
    readFile1 = r'./s2.csv'
    roadClass = '0'

    # city = sys.argv[1]
    # baseFile = sys.argv[2] + '/' + city + '/'
    # readFile = baseFile + city + '_1_100.csv'
    # readFile1 = baseFile + city + '_100-end.csv'
    print(baseFile)
    print(readFile)
    print(readFile1)
    dict = {'0': '40_60', '1': '20_40', '2': '15_25', '3': '15_25', '4': '15_25', '6': '10_20'}

    classRange = dict[roadClass].split("_")
    classRange = list(map(int, classRange))
    print('start')
    # ### 取同根link同分钟不同自由流类别车辆速度数据,全量 大于100m和小于100m
    names = ['car_ff', 'truck_ff', 'min_key', 'linkID', 'roadClass', 'vehicle_type', 'vehicle_num', 'speed_sample', 'key']
    # if readFile != '00':
    dfAll0 = pd.read_csv(readFile, header=None, names=names)
    # if readFile1 != '00':

    dfAll1 = pd.read_csv(readFile1, header=None, names=names)
    # if len(dfAll1) == 0:
    #     dfAll = dfAll1
    # else:
    dfAll = dfAll0.append(dfAll1)

    # dfAll = dfAll0
    dfAll['linkID'] = dfAll['linkID'].astype('str')
    dfAll.head()

    # 过滤速度<40的数据 拥堵
    print('fliter speed <' + str(classRange[0]) + ' start ')
    dfAll35lower = dfAll[
        ((dfAll['vehicle_type'] == 1) & (dfAll['speed_sample'] < classRange[0])) | (dfAll['vehicle_type'] == 0)]
    process(dfAll35lower, '0-' + str(classRange[0]), baseFile)
    print('fliter speed <=' + str(classRange[0]) + '  end ')

    print('fliter speed >' + str(classRange[0]) + ' and speed <=' + str(classRange[1]) + ' start ')
    dfAll35lower = dfAll[((dfAll['vehicle_type'] == 1) & (dfAll['speed_sample'] >= classRange[0]) & (
            dfAll['speed_sample'] < classRange[1])) | (dfAll['vehicle_type'] == 0)]
    process(dfAll35lower, str(classRange[0]) + '-' + str(classRange[1]), baseFile)
    print('fliter speed >' + str(classRange[0]) + ' and speed <=' + str(classRange[1]) + ' end ')

    # 过滤速度>35的数据
    print('fliter speed >' + str(classRange[1]) + ' start ')
    dfAll35lower = dfAll[
        ((dfAll['vehicle_type'] == 1) & (dfAll['speed_sample'] >= classRange[1])) | (dfAll['vehicle_type'] == 0)]
    process(dfAll35lower, classRange[1], baseFile)
    print('fliter speed >' + str(classRange[1]) + ' end ')


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
