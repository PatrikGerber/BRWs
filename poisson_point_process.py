import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sys
sys.setrecursionlimit(60000)

ALPHA = 1.4
BETA = 0.4

class PPP:
	def __init__(self, N):
		self.N = N

	def rejection_sample(self, lower_bound):
		x = -np.inf
		while (x < lower_bound):
			x = stat.levy_stable.rvs(ALPHA, BETA)
		print(np.exp(lower_bound - x))
		if stat.uniform.rvs() < np.exp(lower_bound - x):
			return x
		return self.rejection_sample(lower_bound)

	def points(self):
		I = integrate.quad(lambda x: np.exp(-x)*stat.levy_stable.pdf(x, ALPHA, BETA), 0, np.inf)[0]
		nPoints = stat.poisson.rvs(mu = I)
		lower_bound = upper_bound = 0
		while (nPoints < self.N):
			upper_bound = lower_bound
			lower_bound -= 1
			I = integrate.quad(lambda x: np.exp(-x)*stat.levy_stable.pdf(x, ALPHA, BETA), lower_bound, upper_bound)[0]
			nPoints += stat.poisson.rvs(mu = I)
		return nPoints, lower_bound

	def simulate(self):
		nPoints, lower_bound = self.points()
		print(nPoints, lower_bound)
		return [self.rejection_sample(lower_bound) for _ in range(nPoints)]


# ppp = PPP(4)
# print(ppp.simulate())