from tokenize import Double
from numpy import *;
from sigmoid import sigmoid;

def predict(X, y, theta):
    m = len(X);
    p = zeros((m,1));

    p = sigmoid(dot(X, theta));
    p[where(p>=0.5)] = 1;
    p[where(p<0.5)] = 0;

    accuracy = mean(p==y);

    return accuracy;