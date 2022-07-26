from cv2 import log, multiply
from numpy import *;
from sigmoid import sigmoid;

def costFunctionReg(theta, X, y, lamda):
    grad = zeros(size(theta));
    m = len(X);
    theta_ = theta;
    theta_[0] = 0;
    cost = -sum(multiply(y, log(sigmoid(dot(X, theta))))+multiply((1-y), log(1-sigmoid(dot(X, theta)))))/m+lamda/(2*m)*dot(theta_.T, theta_);
    
    grad = dot(X.T, (sigmoid(dot(X, theta))-y))/m + lamda/m*theta_;

    return cost, grad;
