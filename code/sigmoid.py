from math import exp
from numpy import *;

# 计算h(x)函数

def sigmoid(t):
    result = 1/(1+exp(-t));
    return result;
