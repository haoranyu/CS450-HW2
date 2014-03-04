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
	plt.xlabel("X1 axis")
	plt.ylabel("X2 axis")
	plt.title("Plot for Problem5 a)")
	plt.hold(True)
	plt.gca().set_aspect("equal")

data = make_data()

plot_init(data[0], data[1])
plt.savefig("problem5_a.png")