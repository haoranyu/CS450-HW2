from __future__ import division
import scipy as sp
import scipy.io as sio
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from problem3_a import mgs_qr
from problem3_b import house_qr
from math import pow

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

def least_squares_numpy(X, Y, n):
	array = []
	for n in range(0, n+1):
		tn = []
		for i in X:
			tn.append(i**n)
		array.append(tn)
		A = np.array(array).T
		sol = la.lstsq(A, Y)[0]
	return sol

def least_squares_mgs(X, Y, n):
	array = []
	for n in range(0, n+1):
		tn = []
		for i in X:
			tn.append(i**n)
		array.append(tn)
		A = np.array(array).T

		(Q,R) = mgs_qr(A)
		y = Q.T.dot(Y)
		sol = la.solve(R, y)
	return sol

def least_squares_house(X, Y, n):
	array = []
	for n in range(0, n+1):
		tn = []
		for i in X:
			tn.append(i**n)
		array.append(tn)

		A = np.array(array).T

		(Q,R) = house_qr(A)
		y = Q.T.dot(Y)
		sol = la.solve(R, y)
	return sol

def plot_init(X, Y):
	plt.clf()
	plt.scatter(X, Y)
	plt.xlabel("Months")
	plt.ylabel("Price")
	plt.title("Prices of gasoline over 345 months")
	plt.hold(True)
	

def draw(X, sol):
	n = len(sol) - 1
	x = np.linspace(min(X), max(X), num = 100)
	vals = list()
	for i in x:
		cur_val = 0
		for j in range(0, len(sol)):
			cur_val = cur_val + pow(i, j) * sol[j]
		vals.append(cur_val)
	if(n!=0):
		plt.plot(x, vals, label="Degree %d" % n)
		plt.legend(loc="best")
	plt.show()

def output_sol(sol):
	L = ["a", "b", "c", "d", "e", "f"]
	out = ""
	n = len(sol)
	for j in range(0, n):
		out = out +((j!=0) and " + " or "")+ L[j] +(((n-j-1)!=0) and " * x"+ (((n-j-1)!=1) and "^"+str(n-j-1) or "")  or "")
	print out
	for j in range(0, n):
		print L[j] + " = " + str(sol[n-j-1])
	print ""

Y = np.genfromtxt("Price_of_Gasoline.txt", delimiter="\n")
X = np.zeros(len(Y))

for i in range(len(Y)):
    X[i] = i+1

print "Solution of mgs:"
plot_init(X, Y)
for N in range(1,6):
	sol = least_squares_mgs(np.copy(X),np.copy(Y),N)
	output_sol(sol)
	draw(X, sol)
plt.savefig("problem3_d_mgs.png")

print "Solution of house:"
plot_init(X, Y)
for N in range(1,6):
	sol = least_squares_house(np.copy(X),np.copy(Y),N)
	output_sol(sol)
	draw(X, sol)
plt.savefig("problem3_d_house.png")

print "Solution of lstsq:"
plot_init(X, Y)
for N in range(1,6):
	sol = least_squares_numpy(np.copy(X),np.copy(Y),N)
	output_sol(sol)
	draw(X, sol)
plt.savefig("problem3_d_numpy.png")