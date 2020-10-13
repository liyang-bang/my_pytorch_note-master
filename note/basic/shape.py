#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/23/2020 18:00'
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

'''
超过100种tensor的运算操作，包括转置，索引，切片，数学运算， 线性代数，随机数等，具体访问
https://pytorch.org/docs/stable/torch.html
'''
import torch

#改变形状：如果想改变形状，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
print(z)
#如果是仅包含一个元素的tensor，可以使用.item()来得到对应的python数值
x = torch.randn(1)
print(x)
print(x.item())