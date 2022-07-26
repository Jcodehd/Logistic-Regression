from numpy import * ;
import numpy as np

def mapFeature(X1, X2, degree):
    m = len(X1);
    out = ones((m,1));

    for i in range(1, degree+1):
        for j in range(i+1):
            t = np.array(zeros((m,1)));
            t = (np.array(X1) ** (i-j))*(np.array(X2) ** j);
            out = hstack((out, t));

    return out;
