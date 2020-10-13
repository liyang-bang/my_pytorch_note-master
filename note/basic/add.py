#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/23/2020 17:48'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓     ┏┓
            ┏━┛┻━━━━━┛┻━┓
            ┃     ☃     ┃
            ┃  ┳┛   ┗┳  ┃
            ┃     ┻     ┃
            ┗━┓       ┏━┛
              ┃       ┗━━━━┓
              ┃   神兽保佑  ┣┓
              ┃   永无BUG！ ┣┛
              ┗━┓┓┏━━━━┳┓┏━┛
                ┃┫┫    ┃┫┫
                ┗┻┛    ┗┻┛
"""

from __future__ import print_function
import torch


'''
加法
'''
x = torch.rand(5, 3)
# print(x)
#加法：形式一
y = torch.rand(5, 3)
print(x + y)
#加法：形式二
print(torch.add(x, y))
#加法：给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
#加法：原位/原地操作(in-place）
# 任何一个in-place改变张量的操作后面都固定一个_。例如x.copy_(y)、x.t_()将更改x
# adds x to y
y.add_(x)
print(y)

print(y[:2, 0])

