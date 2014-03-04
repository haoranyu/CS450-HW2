from __future__ import division
import scipy as sp
import scipy.linalg
import numpy as np
import numpy.linalg as la

def randV(seed, m):
	np.random.seed(seed)
	x = np.random.randn(m, 1)
	return x

def invIter( A, x, sigma):
	m, n = A.shape
	As = A - sigma * np.eye(m)
	k = 0
	eps = 10e-12
	delx = 1
	while (delx > eps):
		k+=1
		u = x / la.norm(x)
		x = la.solve(As,u)
		delx = la.norm(abs(u) - abs(x / la.norm(x)))
		lam = np.dot(u.T,x)

	lam = 1. / lam + sigma
	x	= x / la.norm(x)
	return lam, x, k 

A = np.array([[6,2,1], [2,3,1], [1,1,1]])
m, n = A.shape

for ra in range(10):
	x = randV(2, m)
	eigValue, eigVector, time = invIter(np.copy(A), np.copy(x), 2)
	print "eigenvalue: %g" % eigValue
	print "eigenvector:"
	print eigVector
	print "number of iterations: %d" % time