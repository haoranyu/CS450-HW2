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

eigValue = scipy.linalg.eig(np.copy(A))[0]
eigVector = scipy.linalg.eig(np.copy(A))[1].T

print "eigenvalues:"
print eigValue
print "eigenvectors:"
print eigVector

x = randV(2, m)
eigValue_inv, eigVector_inv, time_inv = invIter(np.copy(A), np.copy(x), 2)
eigValue_RQ, eigVector_RQ, time_RQ = RQIter(np.copy(A), np.copy(x))

print "Inverse iteration:"
print "relative error in eigenvalue: " 
print la.norm(eigValue_inv - eigValue[1])/la.norm(eigValue[1])
print "relative error in eigenvector: " 
print la.norm(abs(eigVector_inv.T) - abs(eigVector[1]))/la.norm(eigVector[1])

print "Rayleigh quotient iteration:"
print "relative error in eigenvalue: " 
print  la.norm(eigValue_RQ - eigValue[2])/la.norm(eigValue[2])
print "relative error in eigenvector: " 
print  la.norm(abs(eigVector_RQ.T) - abs(eigVector[2]))/la.norm(eigVector[2])