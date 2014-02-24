from __future__ import division
import scipy as sp
import numpy as np
import numpy.linalg as la

def house(x):
    sigma = np.sqrt(np.dot(x.T,x))[0,0]
    
    v = x.copy()
    if x[0] <= 0:
        v[0] -= sigma

    else:
        v[0] += sigma
    
    v = v/v[0]
    beta = 2./np.dot(v.T,v)[0,0]
    
    return v, beta

def house_qr(A):
    m,n = A.shape
    q = np.eye(m)
    H = np.zeros((m,m))
    for k in range(n):
        v,beta = house(A[k:,k:k+1])
        A[k:,k:] -= beta * np.dot(v, np.dot(v.T,A[k:,k:]))
        H[...] = np.eye(m)
        H[k:,k:] -= beta * np.dot(v,v.T)
        q = np.dot(q,H)
    return q, np.triu(A)