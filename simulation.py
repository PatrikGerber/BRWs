import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

class pointProcess:
	def __init__(self):
		pass

	def simulate(self):
		n = 2
		return stat.rayleigh.rvs(scale = 4, size=n)

class N_BRW:
	def __init__(self, pointProcess, N, startPos = None):
		self.N = N
		self.pointProcess = pointProcess
		self.currPos = startPos if startPos else [0]*N
		self.history = [self.currPos[:]]
		
	def run(self, nGen):
		for gen in range(nGen):
			self.currPos = select(branch(self.currPos, self.pointProcess), self.N)[0]
			self.history += [self.currPos]

# INPUT array of positions and a poinProcess supporting .simulate()
# OUTPUT array of tuples of the form (position, parent index)
def branch(position, pointProcess):
	n = len(position)
	descendants = [None]*n
	for particle in range(n):
		children = pointProcess.simulate()
		nChild = len(children)
		descendants[particle] = zip([position[particle]]*nChild + children, [particle]*nChild)
	return reduce(lambda x,y: x+y, descendants)

# INPUT array of positions of descendants and the number N
# OUTPUT tuple of the next generations and the genealogy after selection
def select(descendants, N):
	indices = np.argsort([d[0] for d in descendants])[-N:]
	nextGen = [descendants[i][0] for i in indices]
	genealogy = [descendants[i][1] for i in indices]
	return (nextGen, genealogy)

N = 10
nGenerations = 100
nBRW = N_BRW(pointProcess(), N)
nBRW.run(nGenerations)
x = reduce(lambda x,y:x+y, [[i]*N for i in range(nGenerations + 1)])
y = reduce(lambda x,y:x+y, nBRW.history)
plt.plot(x, y, 'o')
plt.show()




