#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/23/2020 17:40'
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



#创建一个随机初始化矩阵：
x = torch.rand(5, 3)
print(x)
#构造一个填满0且数据类型为long的矩阵:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
#直接从数据构造张量
print('*'*20)
x = torch.tensor([5.5, 3])
print(x)
#或者根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：
x = x.new_ones(5, 3)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(x)  # 结果size一致


