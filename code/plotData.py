from matplotlib import markers
from numpy import *;
from matplotlib.pyplot import *;


def plotData(X, y):
    m = len(X);

    pos_ = where(y == 1)[0];
    neg_ = where(y == 0)[0];

    fig, ax = subplots(figsize=(6,4));
    ax.scatter(array(X[pos_][:,:1]), array(X[pos_][:,1:]), c='b', marker='o', label='Admitted');
    ax.scatter(array(X[neg_][:,:1]), array(X[neg_][:,1:]), c='r', marker='x', label='no Admitted');
    ax.legend();
    show();
    