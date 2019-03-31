import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

class PointProcess:
	def __init__(self):
		pass

	def simulate(self):
		n = 2
		# return stat.rayleigh.rvs(scale = 4, size=n)
		# return stat.norm.rvs(size = n)
		return stat.cauchy.rvs(size = n)

class N_BRW:
	def __init__(self, pointProcess, N):
		self.N = N
		self.pointProcess = pointProcess
		self.currPos = [0]*N
		self.history = [self.currPos[:]]
		
	def run(self, nGen):
		for gen in range(nGen):
			print('Currently in generation {0}'.format(gen))
			self.currPos = self.select(self.branch())[0]
			self.history += [self.currPos]

	# INPUT array of positions and a poinProcess supporting .simulate()
	# OUTPUT array of tuples of the form (position, parent index)
	def branch(self):
		n = len(self.currPos)
		descendants = [None]*n
		for particle in range(n):
			children = np.array(self.pointProcess.simulate())
			nChild = len(children)
			descendants[particle] = zip(np.array([self.currPos[particle]]*nChild) + children, [particle]*nChild)
		return reduce(lambda x,y: x+y, descendants)

	# INPUT array of positions of descendants and the number N
	# OUTPUT tuple of the next generations and the genealogy after selection
	def select(self, descendants):
		indices = np.argsort([d[0] for d in descendants])[-self.N:]
		nextGen = [descendants[i][0] for i in indices]
		genealogy = [descendants[i][1] for i in indices]
		return (nextGen, genealogy)

# Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] 
# nGens = [i*5 for i in Ns]
Ns = [10]
nGens = [1000]
speeds =[]

for N, nGenerations in zip(Ns, nGens):
	pointProcess = PointProcess()
	nBRW = N_BRW(pointProcess, N)
	nBRW.run(nGenerations)

	time = range(nGenerations + 1)
	rolledTime = reduce(lambda x,y:x+y, [[i]*N for i in range(nGenerations + 1)])

	positions = nBRW.history
	rolledPositions = reduce(lambda x,y:x+y, nBRW.history)

	avgs = np.array(map(np.mean, positions))
	stds = np.array(map(np.std, positions))

	speeds += [avgs[-1]/nGenerations]

	# plt.fill_between(time, avgs - 2 * stds, avgs + 2 * stds, alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')
	plt.plot(rolledTime, rolledPositions, '.')
	plt.show()

# file = open('normal_speeds.txt', 'a')
# file.write(str(Ns) + '\n')
# file.write(str(speeds))
# file.close()







