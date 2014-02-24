from __future__ import division
import scipy as sp
import numpy as np
import numpy.linalg as la

def mgs_qr(A):
	m, n = A.shape
	r = np.zeros((n, n))
	q = np.zeros((m, n))
	for k in range(n):
		r[k, k] = np.linalg.norm(A[:, k])
		q[:, k] = A[:, k] / r[k, k]
		r[k, k + 1:n] = np.dot(q[:, k], A[:, k + 1:n])
		A[:, k + 1:n] = A[:, k + 1:n] - np.outer(q[:, k], r[k, k + 1:n])
	return q, r