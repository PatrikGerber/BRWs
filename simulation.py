import scipy.stats as stat
import numpy as np

class pointProcess:
	def __init__(self):
		pass

	def simulate(self):
		n = 2
		return stat.norm.rvs(size=n)

class N_BRW:
	def __init__(self, pointProcess, N, startPos = None):
		self.N = N
		self.pointProcess = pointProcess
		self.startPos = startPos if startPos else [0]*N
		
	def run(self, nGen):
		self.currPos = self.startPos
		for gen in range(nGen):
			self.currPos = select(branch(self.currPos, self.pointProcess), self.N)[0]
		return self.currPos

def branch(position, pointProcess):
	n = len(position)
	descendants = [None]*n
	for particle in range(n):
		children = pointProcess.simulate()
		nChild = len(children)
		descendants[particle] = zip([position[particle]]*nChild + children, [particle]*nChild)
	return reduce(lambda x,y: x+y, descendants)

def select(descendants, N):
	indices = np.argsort([d[0] for d in descendants])[-N:]
	nextGen = [descendants[i][0] for i in indices]
	genealogy = [descendants[i][1] for i in indices]
	return (nextGen, genealogy)

x = N_BRW(pointProcess(), 2)
print(x.run(100))