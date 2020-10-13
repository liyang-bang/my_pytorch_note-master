#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pytorch_note'
__author__ = 'deagle'
__date__ = '7/24/2020 10:53'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""

import torch

#torch.Tensor 是这个包的核心类。如果设置它的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。

import torch
#创建一个张量并设置requires_grad=True用来追踪其计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)

print(y.grad_fn)


z = y * y * 3
out = z.mean()
print(z, out)

print('-'*50)
#.requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是 False。
a = torch.randn(2, 2)#a = torch.randn(2, 2,requires_grad=True)，就是说默认不会保存追踪历史
print(a)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print('-'*50)
'''
梯度
现在开始进行反向传播，因为 out 是一个标量，因此 out.backward() 和 out.backward(torch.tensor(1.)) 等价。
'''

out.backward()
print(x.grad)




if __name__ == '__main__':
    print()