from __future__ import division
import scipy as sp
import scipy.io as sio
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def make_data(dims=2, npts=3000):
	np.random.seed(13)
	
	mix_mat = np.random.randn(dims, dims)
	mean = np.random.randn(dims)
	return np.dot(mix_mat, np.random.randn(dims, npts)) + mean[:, np.newaxis]

def plot_init(X, Y):
	plt.clf()
	plt.scatter(X, Y)
	plt.xlabel("Y'1 axis")
	plt.ylabel("Y'2 axis")
	plt.title("Plot for Problem5 d)")
	plt.hold(True)
	plt.gca().set_aspect("equal")

def get_Y_p(data):
	x_apx = np.copy(data)
	N = len(data[0])
	sum_xi = np.array([0,0])
	for i in range(N):
		sum_xi += x_apx.T[i]
	u = 1/N * sum_xi.T
	for i in range(N):
		x_apx.T[i] -= u
	Y = 1/np.sqrt(N-1) * x_apx
	
	U, s, V = la.svd(Y,1)
	m, n = Y.shape
	sigma = np.zeros((m, n))
	sigma[0:m, 0:m] = np.diag(s)
	sigma[m-1,m-1] = 0
	Y_p = np.dot(U, np.dot(sigma, V))
	Y_p *= np.sqrt(N-1)
	for i in range(N):
		Y_p.T[i] += u
	return Y_p

def principal_component(data):
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
	return np.dot(U, sigma)

def draw_principal_component(data):
	x_apx = np.copy(data)
	N = len(data[0])
	sum_xi = np.array([0,0])
	for i in range(N):
		sum_xi += x_apx.T[i]
	mean = sum_xi.T/N
	prin = principal_component(np.copy(data))
	for i in range(len(prin.T)):
		aw = prin.T[i]
		if (aw[1]!=0 or aw[0]!=0):
			plt.arrow(mean[0], mean[1], aw[0], aw[1], color="red", head_width=0.05, head_length=0.1)


data = make_data()
Y_p = get_Y_p(np.copy(data))
plot_init(Y_p[0], Y_p[1])
draw_principal_component(Y_p)
plt.savefig("problem5_d.png")
