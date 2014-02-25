from __future__ import division
import scipy as sp
import scipy.linalg
import numpy as np
import numpy.linalg as la

def randV(seed, m):
	np.random.seed(seed)
	x = np.random.randn(m, 1)
	return x

def invIter(A, x):
	sigma = 2
	P, L, U = scipy.linalg.lu(A - sigma * np.eye(m))
	eps = np.spacing(1)
	delx = 1
	i = 0
	while (delx > eps):
		i += 1
		y = la.solve(U, la.solve(L,x))
		y = y / la.norm(y)
		delx = la.norm(abs(y) - abs(x))
		x = y
	y = np.dot(A,x)
	return x, la.norm(y), i

A = np.array([[6,2,1], [2,3,1], [1,1,1]])
m, n = A.shape

for ra in range(10):
	x = randV(2, m)
	eigVector, eigValue, time = invIter(np.copy(A), np.copy(x))
	print "eigenvalue: %g" % eigValue
	print "eigenvector:"
	print eigVector
	print "number of iterations: %d" % time