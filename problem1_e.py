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

def problem_e(n):
	A = rand_spd(n)
	L = cholesky(np.copy(A), n)
	det = np.prod(np.diag(L)) * np.prod(np.diag(L.T))
	det_true = la.det(A)
	error = (det - det_true) / (det_true)

	print "the size of the matrix n: \t%g" % n
	print "numpy.linalg.det(A): \t%g" % det_true
	print "my computed determinant: \t%g" % det
	print "relative error: \t%g\n" % error

problem_e(5)
problem_e(10)
problem_e(100)