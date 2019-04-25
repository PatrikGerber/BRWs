import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import expi
import sys
from pynverse import inversefunc
sys.setrecursionlimit(60000)

# THe class representing the N-BRW driven by the Cauchy PPP. 
class CPP_nBRW:
	# INPUT the number N
	def __init__(self, N):
		self.N = N
		self.currPos = [0]*N
		self.history = [self.currPos[:]]

	# INPUT nGen, the number of generations to run the N-BRW for. 
	# OUTPUT stores the N-BRW in the array self.history
	def run(self, nGen):
		for gen in range(nGen):
			print('Currently in generation {0}'.format(gen))
			self.currPos = self.branch_and_select()
			self.history += [self.currPos]

	# The rest of the class is made up of internal methods 
	# required to do the calculations

	
	def unnormalized_cdf(self, x, lower_bound):
		return self.primitive(x) - self.primitive(lower_bound)

	def primitive(self, x):
		if x == np.inf:
			return np.real(self.N * (np.exp(1j) + np.exp(-1j)) / 2)
		return np.real(sum([1j * np.exp(-1j) * (np.exp(2j)*expi(-(x-x0)-1j) 
					- expi(1j - (x-x0))) / (2*np.pi) for x0 in self.currPos]))

	def inverse_sample(self, lower_bound, upper_bound, normalizing_constant):
		cdf = lambda x: self.unnormalized_cdf(x, lower_bound) 
						/ normalizing_constant
		return inversefunc(cdf, stat.uniform.rvs(), 
							domain=[lower_bound, 
									min(upper_bound, lower_bound + 20)], 
							image=[0,1])

	def simulate(self):
		normalizing_constant = max(self.unnormalized_cdf(np.infty, 0), 0)
		nPoints = stat.poisson.rvs(mu = normalizing_constant)
		positions = [self.inverse_sample(0, np.inf, normalizing_constant) 
					 for _ in range(nPoints)]
		lower_bound = upper_bound = 0
		while (nPoints < self.N):
			upper_bound = lower_bound
			lower_bound -= 1
			normalizing_constant = max(
						self.unnormalized_cdf(upper_bound, lower_bound), 0)
			newPoints = stat.poisson.rvs(mu = normalizing_constant)
			positions += [self.inverse_sample(lower_bound, 
											  upper_bound, 
											  normalizing_constant) 
						  for _ in range(newPoints)]
			nPoints += newPoints
		return positions

	def branch_and_select(self):
		newPositions = sorted(self.simulate())
		return newPositions[-self.N:]

# The code below runs the simulation for 1000 generations with N=100. 
N = 100
nGenerations = 1000

nBRW = CPP_nBRW(N)
nBRW.run(nGenerations)

time = range(nGenerations + 1)
rolledTime = reduce(lambda x,y:x+y, [[i]*N for i in range(nGenerations + 1)])

positions = nBRW.history
rolledPositions = reduce(lambda x,y:x+y, nBRW.history)
file = open('N{0}_nGen{1}'.format(N, nGenerations), 'a')
file.write(str(positions))

avgs = np.array(map(np.mean, positions))
stds = np.array(map(np.std, positions))

plt.plot(rolledTime, rolledPositions, '.')
plt.show()
