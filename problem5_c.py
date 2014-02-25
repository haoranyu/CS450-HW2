from __future__ import division
import scipy as sp
import numpy as np
import numpy.linalg as la

def make_data(dims=2, npts=3000):
	np.random.seed(13)
	
	mix_mat = np.random.randn(dims, dims)
	mean = np.random.randn(dims)
	return np.dot(mix_mat, np.random.randn(dims, npts)) + mean[:, np.newaxis]

def get_Y(data):
	x_apx = np.copy(data)
	N = len(data[0])
	sum_xi = np.array([0,0])
	for i in range(N):
		sum_xi += x_apx.T[i]
	u = sum_xi.T/N
	for i in range(N):
		x_apx.T[i] -= u
	Y = x_apx/np.sqrt(N-1)
	
	U, s, V = la.svd(Y,1)
	m, n = Y.shape
	sigma = np.zeros((m, n))
	sigma[0:m, 0:m] = np.diag(s)
	Y_apx = np.dot(U, np.dot(sigma, V))
	return Y_apx, Y

data = make_data()
Y_apx, Y = get_Y(np.copy(data))

print "relative error: %g" % (la.norm(Y_apx - Y)/la.norm(Y))
