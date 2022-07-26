from numpy import *;
from costFunction import costFunction;

def gradientDescent(X, y, theta, alpha, iters):
    m = len(X);
    for i in range(iters):
        cost, grad = costFunction(X, y, theta);
        theta = theta - alpha*grad;

    return cost, theta;
