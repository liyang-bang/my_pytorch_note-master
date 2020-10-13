import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import sys


def groupData(df) :
    ps = dict(list(df.groupby(['linkID',])))

    for key in ps:
        dfa = ps[key]
        mylist = dfa['min_key'].values.tolist()
        if len(set(mylist)) > 200:
            print(key)




def getCar_truck_key(a,b):
    return str(a) + '_' + str(b)


def getLinkID(a, b):
    if str(b).startswith('-'):
        return str(a) + "1" + str(b)[1:]
    return str(a) + '0' + str(b)



def main():
    # names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
    city = '350100'
    dfEquals0 = pd.read_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/' + city + ' - 副本/diff-smallEqualBig-Link-0.csv',
                            header=None)
    dfEquals1 = pd.read_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/' + city + ' - 副本/diff-smallEqualBig-Link-1.csv',
                            header=None)
    # dfEquals0 = pd.read_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/'+city+' - 副本/diff-smallHighBig-Link-0.csv', header=None)
    # dfEquals1 = pd.read_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/'+city+' - 副本/diff-smallHighBig-Link-1.csv', header=None)
    dfEqualsAll = dfEquals0.append(dfEquals1)
    dfEqualsAll['linkId'] = dfEqualsAll.apply(lambda row: getLinkID(row[0], row[1]), axis=1).astype('str')
    print(len(dfEqualsAll))

    names = ['car_ff', 'truck_ff', 'min_key', 'linkID', 'vehicle_type', 'vehicle_num', 'speed_sample', 'key']
    dfAll = pd.read_csv('E:/CennaviWorkSpace/Jupyter/freeFlowAna/' + city + '_1_100.csv', header=None, names=names)
    dfAll['linkID'] = dfAll['linkID'].astype('str')
    dfAll['car_truck_key'] = dfAll.apply(lambda row: getCar_truck_key(row[0], row[1]), axis=1).astype('str')
    print(dfAll.head())

    # li = dfEqualsAll['linkId'].values.tolist()
    # df = dfAll[dfAll['linkID'].isin(li)]


if __name__ == '__main__':
    main()








