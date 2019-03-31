import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import expi
import sys
from pynverse import inversefunc
# sys.setrecursionlimit(60000)

class CPP:
	def __init__(self, N):
		self.N = N

	def inverse_sample(self, lower_bound, upper_bound, I):
		cdf = lambda x: integrate.quad(lambda y: np.exp(-y)*stat.cauchy.pdf(y), lower_bound, x)[0] / I
		# grid = np.arange(lower_bound, 10, 0.1)
		# plt.plot(grid, [cdf(x) for x in grid])
		# plt.show()
		return inversefunc(cdf, stat.uniform.rvs(), domain=[lower_bound, min(upper_bound, 50)], image=[0,1])

	def simulate(self):
		I = integrate.quad(lambda x: np.exp(-x)*stat.cauchy.pdf(x), 0, np.inf)[0]
		nPoints = stat.poisson.rvs(mu = I)
		positions = [self.inverse_sample(0, np.inf, 0.197814) for _ in range(nPoints)]
		lower_bound = upper_bound = 0
		while (nPoints < self.N):
			upper_bound = lower_bound
			lower_bound -= 1
			I = integrate.quad(lambda x: np.exp(-x)*stat.cauchy.pdf(x), lower_bound, upper_bound)[0]
			newPoints = stat.poisson.rvs(mu = I)
			positions += [self.inverse_sample(lower_bound, upper_bound, I) for _ in range(newPoints)]
			nPoints += newPoints
		return positions

# cpp = CPP(50)
# cpp.inverse_sample(-10, 91.3947)
# cpp.inverse_sample(0, 0.197814)
# print(cpp.simulate())