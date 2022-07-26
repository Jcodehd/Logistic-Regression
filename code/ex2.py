from math import sqrt
from telnetlib import PRAGMA_HEARTBEAT
from cv2 import multiply
from numpy import *
import numpy as np;
from pandas import *;
from plotData import plotData;
from mapFeature import mapFeature;
from costFunctionReg import costFunctionReg;
from gradientDescentReg import gradientDescentReg;
from predict import predict;
import matplotlib.pyplot as plt;
import mes 


# 读取数据
data = mat(read_table('Machine Learning\Logistic Regression\ex2data2.txt', sep=',', header=None));
X = data[:,:2];
y = data[:,2:];

# 画出数据散点图 
plotData(X, y);

# 特征映射
X = mapFeature(X[:,:1],X[:,1:], 6);
# 样本数量
m = X.shape[0];
# 特征
n = X.shape[1];

# 初始化theta, lamda
theta = zeros((n, 1));
lamda = 1;

# 计算损失
cost, grad= costFunctionReg(theta, X, y, lamda);

# 梯度下降

cost_, theta = gradientDescentReg(theta, X, y, 7, 200000);
print(cost_);
print(theta);


print('训练模型的精度为: ', predict(X, y, theta));










