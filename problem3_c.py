from __future__ import division
import scipy as sp
import numpy as np
import numpy.linalg as la

from problem3_a import mgs_qr
from problem3_b import house_qr

def result(m,n):
	np.random.seed(0)
	A = np.random.randn(m, n)
	print "matrix shape: \t"
	print A.shape

	Q1, R1 = mgs_qr(np.copy(A))
	error = la.norm(np.dot(Q1,R1)-A)/la.norm(A)
	print "mgs relative error: \t%g" % error

	Q2, R2 = house_qr(np.copy(A))
	error = la.norm(np.dot(Q2,R2)-A)/la.norm(A)
	print "house relative error: \t%g" % error

	cond = la.cond(A)
	print "condition number of A: \t%g" % cond


result(5,5)
result(10,10)
result(100,80)