#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/28/2020 17:57'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import torch


def main():
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    print(x,y)
    print(x+y)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])