import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

# The class representing a Point process with n points. 
class PointProcess:
	# INPUT number of particles in each realisation. n=2 corresponds 
	# 		binary N-BRWs.  
	def __init__(self, n = 2):
		self.n = n

	# INPUT None
	# OUTPUT returns n i.i.d. samples from the offspring distribution 
	def simulate(self):
		# return stat.norm.rvs(size = self.n)
		# return stat.cauchy.rvs(size = self.n)
		return stat.bernoulli.rvs(0.75, size = self.n)

# The class representing the N-BRW. 
class N_BRW:
	# INPUT Point process that drives the N-BRW, the number N. 
	def __init__(self, pointProcess, N):
		self.N = N
		self.pointProcess = pointProcess
		self.currPos = [0]*N
		self.history = [self.currPos[:]]
	
	# INPUT nGen, the number of generations to run the N-BRW for
	# OUTPUT stores the N-BRW in the array self.history
	def run(self, nGen):
		for gen in range(nGen):
			print('Currently in generation {0}'.format(gen))
			self.currPos = self.select(self.branch())[0]
			self.history += [self.currPos]

	# INPUT array of positions and a poinProcess object supporting .simulate()
	# OUTPUT array of tuples of the form (position, parent index)
	def branch(self):
		n = len(self.currPos)
		descendants = [None]*n
		for particle in range(n):
			children = np.array(self.pointProcess.simulate())
			nChild = len(children)
			descendants[particle] = zip(np.array([self.currPos[particle]]*nChild) 
										+ children, [particle]*nChild)
		return reduce(lambda x,y: x+y, descendants)

	# INPUT array of positions of descendants and the number N
	# OUTPUT tuple of the next generations and the genealogy after selection
	def select(self, descendants):
		indices = np.argsort([d[0] for d in descendants])[-self.N:]
		nextGen = [descendants[i][0] for i in indices]
		genealogy = [descendants[i][1] for i in indices]
		return (nextGen, genealogy)


# The code below runs the simulation for 50 generations with N=10. 
nGenerations = 50
N = 10

pointProcess = PointProcess()
nBRW = N_BRW(pointProcess, N)
nBRW.run(nGenerations)

time = range(nGenerations + 1)
rolledTime = reduce(lambda x,y:x+y, [[i]*N for i in range(nGenerations + 1)])

positions = nBRW.history
rolledPositions = reduce(lambda x,y:x+y, nBRW.history)

avgs = np.array(map(np.mean, positions))
stds = np.array(map(np.std, positions))


plt.rcParams.update({'font.size': 18})
plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(rolledTime, rolledPositions, '.')
plt.ylabel('Position')
plt.xlabel('Generation')
plt.show()