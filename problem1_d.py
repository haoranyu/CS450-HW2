from __future__ import division
import scipy as sp
import numpy as np
import numpy.linalg as la

def cholesky(A, n):
	L = np.zeros((n,n))

	for i in range(n):
		for k in range(i+1):
			temp = sum(L[i][j] * L[k][j] for j in range(k))
			if(i==k):
				L[i][k] = np.sqrt(A[i][i] - temp )
			else:
				L[i][k] = (1.0 / L[k][k] * (A[i][k] - temp))
	return L

def rand_spd(n):
	np.random.seed(0)
	A = np.random.rand(n,n)
	return np.dot(A, A.T)

n = 20
A = rand_spd(n)
L = cholesky(np.copy(A), n)

error = la.norm(np.dot(L, L.T) - A) / la.norm(A)
print "relative error: \t%g" % error

cond = la.cond(A)
print "condition number of A: \t%g" % cond