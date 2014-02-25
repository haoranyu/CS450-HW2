from __future__ import division
import scipy as sp
import scipy.linalg
import numpy as np
import numpy.linalg as la

def randV(seed, m):
	np.random.seed(seed)
	x = np.random.randn(m, 1)
	return x

def RQIter( A, x):
	m, n = A.shape
	I = np.eye(m)
	k = 0
	eps = 10e-12
	delx = 1
	while (delx > eps):
		k+=1
		u = x/la.norm(x)
		lam = np.dot(u.T,np.dot(A,u))
		x = la.solve(A-lam*I,u)
		delx = la.norm(abs(u) - abs(x / la.norm(x)))
	u = x/la.norm(x)
	lam = np.dot(u.T,np.dot(A,u))
	x	= x/la.norm(x,2)
	return lam, x, k

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

x = np.array([[1],[4],[2]])
eigValue_inv, eigVector_inv, time_inv = invIter(np.copy(A), np.copy(x), 2)
eigValue_RQ, eigVector_RQ, time_RQ = RQIter(np.copy(A), np.copy(x))

print "Inverse iteration:"
print "Eigenvalue"
print eigValue_inv
print "Eigenvector"
print eigVector_inv
print "Number of iterations %d\n" % time_inv
print "Rayleigh quotient iteration:"
print "Eigenvalue"
print eigValue_RQ
print "Eigenvector"
print eigVector_RQ
print "Number of iterations %d\n" % time_RQ