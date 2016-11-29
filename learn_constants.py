from graph import *
import sys

# Beginning with the values suggested by Dorigo et al, this program
# searches for the optimal values for the various constants in the
# algorithm.  It focuses on the medium size (29 node) graph.

# number of times to try the algorithm for each set of constants
runs = 5

# graph, number of ants, iterations
g = Graph('test_data/western_sahara_29.txt')
numAnts = 29
iterations = 8

def LearnElitistAntSystemConstants():
	betas = [i for i in range(3, 8)]
	decays = [0.1 * i for i in range(1, 10)]
	elitisms = [i for i in range(3, 8)]

	# list of (cost, [bestBeta, bestDecay, bestElitism])
	results = []

	for b in betas:
		for d in decays:
			for e in elitisms:
				# append average solution weight over number of runs
				sumOfCosts = 0.
				for i in range(runs):
					s = AntSystem(g, numAnts, iterations, beta=b, 
						decay=d, elitist=True, elitism=e)
					sumOfCosts += s.solve().length

				results.append( (sumOfCosts / runs, [b, d, e]) )

	# return top ten sets of constants
	return sorted(results, key=lambda x: x[0])[:10]

print '\nElitist Ant System'
print 'cost\tbeta\tdecay\telitism'
print '----------------------------------'
results = LearnElitistAntSystemConstants()
for entry in results:
	print str(int(round(entry[0]))) + '\t' + str(entry[1][0]) + '\t' + str(round(entry[1][1], 1)) + '\t' + str(entry[1][2])
print '----------------------------------'
print 'DONE\n'


def LearnAntColonySystemConstants():
	betas = [i for i in range(1, 5)]
	exploitProbs = [0.1 * i for i in range(1, 10)]
	decays = exploitProbs

	results = []

	for b in betas:
		for e in exploitProbs:
			for d in decays:
				# append average solution weight over number of runs
				sumOfCosts = 0.
				for i in range(runs):
					s = AntColonySystem(g, numAnts, iterations, 
						beta=b, exploitProb=e, decay=d)
					sumOfCosts += s.solve().length

				results.append( (sumOfCosts / runs, [b, e, d]) )

	return sorted(results, key=lambda x: x[0])[:10]

# print '\nAnt Colony System'
# print 'cost\tbeta\texPrb\tdecay'
# print '----------------------------------'
# results = LearnAntColonySystemConstants()
# for entry in results:
# 	print str(int(round(entry[0]))) + '\t' + str(entry[1][0]) + '\t' + str(round(entry[1][1], 1)) + '\t' + str(round(entry[1][2], 1))
# print '----------------------------------'
# print 'DONE\n'


