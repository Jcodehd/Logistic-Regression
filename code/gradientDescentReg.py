from numpy import *;
from costFunctionReg import costFunctionReg;

def gradientDescentReg(theta, X, y, alpha, iters):
    m = len(X);
    for i in range(iters):
        cost, grad = costFunctionReg(theta, X, y, 1);

        theta = theta - alpha*grad;

    return cost, theta;