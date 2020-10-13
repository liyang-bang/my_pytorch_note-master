#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/31/2020 11:36'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import numpy as np

def test():
    a = [1, 1, 2]
    b = [1, 0, 2]
    arr1 = np.array(a)
    arr2 = np.array(b)
    if  (arr2 == 1).any():
        print('arr2 havg 1')

    if (arr1 != 0).all():
        print('arr1 dont hav 1')


def main():
    test()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])