from cv2 import log, multiply
from numpy import *;
from sigmoid import sigmoid;

def costFunction(X, y, theta):
    # 初始化 损失、梯度
    m = len(X);
    grad = zeros(size(theta));
    cost = 0;
    cost = -sum(multiply(y, log(sigmoid(dot(X, theta))))+multiply((1-y), log(1-sigmoid(dot(X, theta)))))/m;

    grad = dot(X.T, (sigmoid(dot(X, theta))-y))/m;

    return cost, grad;