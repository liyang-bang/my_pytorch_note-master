#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/31/2020 11:18'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime

from sklearn.metrics import f1_score


def f1():
    y_true = [0,0,2,0,2,0,2,1,0,0,2,2,0,0,0,1,2,0,1,2,0,2,0,0,1,0,0,0,0,0,0,0]
    y_pred  = [0,0,1,1,0,0,1,2,2,0,0,1,0,0,2,1,2,1,0,2,2,0,2,1,2,2,2,0,2,0,0,2]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]

    print('-'*20)
    print(f1_score(y_true, y_pred, average='macro'))  # 0.26666666666666666
    print(f1_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
    print(f1_score(y_true, y_pred, average='weighted'))  # 0.26666666666666666

def f2():
    y_true = [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred  = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print('-'*20)
    print(f1_score(y_true, y_pred, average='macro'))  # 0.26666666666666666
    print(f1_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
    print(f1_score(y_true, y_pred, average='weighted'))  # 0.26666666666666666


def f3():
    y_true = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred =    [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]

def f4():
    y_true = [1, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 1, 1]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]
    print(f1_score(y_true, y_pred, average=None))  # [0.8 0. 0. ]


def main():
    # f1()
    # print('2-' * 20)
    # f2()
    # print('3-' * 20)
    # f3()
    f4()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])