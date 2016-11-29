import time
import random
import copy
import numpy as np
import math
import sys
from scipy.spatial.distance import pdist, squareform

# Graph class
class Graph:
	# Initialize using textfile of adjacency matrix and list of points
	def __init__(self, textPoints, textMatrix=None):
		self.matrix = []
		self.points = []

		# Parse list of points from text file
		pointsFile = open(textPoints, 'r')
		while True:
			line = pointsFile.readline().split()
			if not line:
				break
			self.points.append([float(entry) for entry in line])

		if not textMatrix:
			# populate adjacency matrix
			self.matrix = squareform(pdist(np.array(self.points), 'euclidean'))
		else:
			matrixFile = open(textMatrix, 'r')
			# Parse adjacency matrix from text file
			while True:
				line = matrixFile.readline().split()
				if not line:
					break
				self.matrix.append([float(entry) for entry in line])
			

		# Number of nodes
		self.numNodes = len(self.points)

# Path class
# A path is an array A where A[i] is the node after i in the path
class Path:
	def __init__(self, graph):
		# -1 signifies a path that hasn't been chosen yet
		self.graph = graph
		self.path = [-1 for p in graph.points]
		self.numNodes = len(self.path)
		self.length = 0

	# Adds a path step from i to j if there is no step already
	def addStep(self, i ,j):
		if self.path[i] == -1:
			self.path[i] = j
			self.length += self.graph.matrix[i][j]

	def cost(self):
		pathCost = 0
		for i in self.path:
			pathCost += self.graph.matrix[i][self.path[i]]
		return pathCost

# Solver class
class Solver:
	def __init__(self, g):
		self.graph = g

	# Solve returns a path
	def solve(self):
		return Path(self.graph)

# Finds a Greedy solution, jumping to the nearest unvisited
# point for each step in the path
class Greedy(Solver):
	def solve(self):
		startTime = time.time()
		p = Path(self.graph)
		length = 0
		visited = set([0])
		current = 0
		while length < p.numNodes:
			nextNodes = [j for j in range(p.numNodes) if j not in visited and j != current]
			if nextNodes:
				bestNode = min(nextNodes, key=lambda n: self.graph.matrix[current][n])
				p.addStep(current, bestNode)
				visited.add(bestNode)
				current = bestNode
			# Last step must take us back to starting point
			else:
				p.addStep(current, 0)
			length += 1

		endTime = time.time()

		# Runtime in ms.
		self.time = (endTime - startTime) * 1000
		return p

# Finds a random path
class Random(Solver):
	def solve(self):
		startTime = time.time()
		p = Path(self.graph)
		length = 0
		visited = set([0])
		current = 0
		while length < p.numNodes:
			nextNodes = [j for j in range(p.numNodes) if j not in visited and j != current]
			if nextNodes:
				nextNode = random.sample(nextNodes, 1)[0]
				p.addStep(current, nextNode)
				visited.add(nextNode)
				current = nextNode
			# Last step must take us back to starting point
			else:
				p.addStep(current, 0)
			length += 1

		endTime = time.time()

		# Runtime in ms.
		self.time = (endTime - startTime) * 1000
		return p

# Possible extension: ElitistAntSystem (p. 44)
# Possible extension: AntColonySystem (using local search)

# Time Complexity: O(t * n^2 * m)
	# t = number of iterations
	# n = number of cities
	# m = number of ants
class AntSystem(Solver):
	# Variables
	# distance (between city i and city j): d_ij
	# visibility (heuristic desiribility of edge ij): v_ij = 1/d_ij
	# pheromone strength (learned desiribility of edge ij): pher_ij
		# initially set to some small, positive constant c
	# transition rule (probability that ant takes edge ij):
		# p_if = (pher_ij ** a) * (v_ij ** b) / sum((pher_il ** a) * (v_il ** b)) for all unvisited neighbors l to i
	# learned change in pheromone strength: delta_ij = C/tour_length if edge ij in tour, 0 o/w
		# C is a constant that should be a reasonable value for a good tour cost (for example,
		# the tour cost found by the Greedy solver)
	# decay of pheromone strength over time: decay

	# Update step (at the end of a tour):
		# pher_ij <- (1 - decay) * pher_ij + delta_ij  

	# Begin with one ant on each of the cities
	# Each ant is independent of the others, but they all rely on the pheromone strengths
	# from previous iterations, which have been updated by all ants.


	# initial pheromone strength
	initPherStrength = 1e-6

	def __init__(self, g, numAnts, iterations, alpha=1, beta=5, decay=0.5, elitist=False, elitism=5):
		self.graph = g
		self.pherStrengths = [[AntSystem.initPherStrength for col in row] for row in g.matrix]
		self.numAnts = numAnts
		self.iterations = iterations
		self.greedyCost = 0

		# scaling factors for pheromone strength, visibility (inverse distance), pheremone
		# decay over time
		self.alpha = alpha
		self.beta = beta
		self.decay = decay

		self.elitist = elitist
		if elitist:
			# amount of elitist ant reinforcement
			self.elitism = elitism

			self.bestPath = None
			self.bestPathCost = sys.maxint

		# timing segments
		self.totalTime = 0

	def solve(self):
		totalStartTime = time.time()

		# get the cost of a greedy solution, for use later on.
		path = Greedy(self.graph).solve()
		self.greedyCost = path.cost()
		
		for t in range(self.iterations):
			# maintain list of points from which ants have started a tour
			availableStartingPoints = set([i for i in range(self.numAnts)])

			# get current edge desirabilities for this iteration
			edgeDesirabilities = [[self.computeEdgeDesirability(row, col, self.pherStrengths) for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]

			# each ant is assigned a starting point and finds a tour from that starting point,
			# updating pheromone strength according to optimality of that path
			for k in range(self.numAnts):
				# replenish set of starting points if necessary (if numAnts > numNodes)
				if not availableStartingPoints:
					availableStartingPoints = set([i for i in range(self.numAnts)])

				# choose random starting point
				startingPoint = random.sample(availableStartingPoints, 1)[0]
				availableStartingPoints.remove(startingPoint)

				# each ant works off its own copy of edgeDesirabilities, so that
				# ants move independently at each time iteration
				path = self.findPath(copy.copy(edgeDesirabilities), startingPoint)

				# update bestPath if we need to
				cost = path.cost()
				if self.elitist and cost < self.bestPathCost:
					self.bestPath = path
					self.bestPathCost = cost

				# each ant updates pheromone strengths (a global variable), so that
				# in the next iteration, all ants work with updated pheromones.
				self.updatePheromoneStrengths(path)

		# return best path according to stored pheromone strengths
		bestPath = self.computeBestPath()

		totalEndTime = time.time()
		self.totalTime = (totalEndTime - totalStartTime) * 1000

		return bestPath

	def computeEdgeDesirability(self, row, col, pherStrengths):
		# an edge's desirability is the product of the pheromone strength and the weight
		# of the edge, both scaled by alpha and beta.

		# the desirability of a self-loop is 0.
		if row == col:
			return 0.0
		return (pherStrengths[row][col] ** self.alpha) * ((1 / self.graph.matrix[row][col]) ** self.beta)

	def normalizeRow(self, row):
		rowSum = sum(row)
		row = [i / rowSum for i in row]
		return row

	# finds a path based on current pheromone strengths
	def findPath(self, edgeDesirabilities, startingPoint):
		p = Path(self.graph)
		length = 0
		current = startingPoint

		edgeDesirabilities = [[0 if col == current else edgeDesirabilities[row][col] for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]
		visited = set([current])
		while length < p.numNodes - 1:
			# Normalize each row in edgeDesirabilities, as row i
			# represents the probabilities with which the ant should move
			# from city i to every other potential city.
			# Remove visited nodes right here, to avoid having to re-sample
			# if the sampled node turns out to be visited.
			probs = []
			unvisited = []
			for i, prob in enumerate(edgeDesirabilities[current]):
				if i not in visited:
					unvisited.append(i)
					probs.append(prob) 
			probs = self.normalizeRow(probs)
			
			neighbor = np.random.choice(unvisited, 1, p=probs)[0]
			visited.add(neighbor)
			p.addStep(current, neighbor)

			# make sure that this neighbor can never be visited again
			# by setting the probability that any city visits it to be 0.
			edgeDesirabilities = [[0 if col == neighbor else edgeDesirabilities[row][col] for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]

			current = neighbor
			length += 1

		# add the final edge back to the starting point
		p.addStep(current, startingPoint)
		return p

	# updates pheromone strengths based on the strength of a path
	def updatePheromoneStrengths(self, path):
		for row in range(len(self.pherStrengths)):
			for col in range(len(self.pherStrengths)):
				# if the edge row->col is not in the path, don't update the pheromone strength.
				# because path[i] gives the node connected to node i, if
				# path[row] == col, there is an edge from row to col.
				delta = self.greedyCost / path.cost() if path.path[row] == col else 0
				self.pherStrengths[row][col] = (1 - self.decay) * self.pherStrengths[row][col] + delta

				if self.elitist:
					eliteReinforcement = self.greedyCost / self.bestPathCost if self.bestPath.path[row] == col else 0
					self.pherStrengths[row][col] += self.elitism * eliteReinforcement

	# computes the best path given existing pheromone trails
	def computeBestPath(self):
		bestPathCost = sys.maxint
		bestPath = None
		for start in range(self.graph.numNodes):
			# Choose a path deterministically, always picking the edge with the 
			# strongest pheromone.
			p = Path(self.graph)
			length = 0
			visited = set([0])
			current = 0
			while length < p.numNodes:
				nextNodes = [j for j in range(p.numNodes) if j not in visited and j != current]
				if nextNodes:
					bestNode = max(nextNodes, key=lambda n: self.pherStrengths[current][n])
					p.addStep(current, bestNode)
					visited.add(bestNode)
					current = bestNode
				# Last step must take us back to starting point
				else:
					p.addStep(current, 0)
				length += 1

			cost = p.cost()
			if cost < bestPathCost:
				bestPath = p 
				bestPathCost = cost
		return bestPath

class AntColonySystem(Solver):
	# Variables
	# distance (between city i and city j): d_ij
	# visibility (heuristic desiribility of edge ij): vis_ij = 1/d_ij
	# pheromone strength (learned desiribility of edge ij): pher_ij
	# candidate list (list of preferred cities to be visited from a given city): cand_i
		# set initial pheremone strength pher_0 = 1/(n * (cost of greedy solution))

	# transition rule:
		# let q = a random variable on Unif(0,1), exploit_prob a tunable parameter on [0,1]
		# the next j to move to is given by
		# city with the max value of (pher_ij) * (vis_ij ** b) over unvisited neighbors, if q <= exploit_prob
		# sample with probs (pher_ij) * (vis_ij ** b) / sum((pher_il) * (vis_il ** b)) over adjacent
		# neighbors as before, if q > exploit_prob

	# pheremone trail update rule:
		# in contrast to ant system, only the ant who has found the best solution so far will update
		# the pheromone strengths 
		# furthermore, the pheromone strength update only applies to edges that are part of the best
		# tour found so far
		# learned change in pheromone strength: delta_ij = 1/(cost of best tour)
		# update step (at the end of a tour) is only relevant for the edges ij in the best tour:
			# pher_ij <- (1 - decay) * pher_ij + delta_ij

	# local pheromone trail update rule:
		# every ant lessens the amount of pheromone on it's trail.  this has the effect of promoting
		# exploration of not-yet visited edges
		# pher_ij <- (1 - decay) * pher_ij + decay * pher_0

	# candidate list:
		# contains the k closest cities ordered by distance (where k is a tunable parameter)
		# if there are unvisited cities in the candidate list, the next city j is chosen from the
		# candidate list as described above. Otherwise, j is the closest unvisited city

	# helper to compute candidates (as edges)
	def getCandidates(self, g, candidatesPerNode):
		candidates = []
		for row in g.matrix:
			sortedRow = sorted(row)
			candidates.append(sortedRow[:candidatesPerNode])
		return candidates

	def __init__(self, g, numAnts, iterations, beta=2, exploitProb=0.9, decay=0.1):
		self.graph = g
		self.pherStrengths = [[AntSystem.initPherStrength for col in row] for row in g.matrix]
		self.numAnts = numAnts
		self.iterations = iterations

		# compute cost of greedy solution
		self.greedyCost = Greedy(self.graph).solve().cost()

		# initial pheromone strength now depends on the graph, so it must be an instance var
		self.initPherStrength = 1 / (g.numNodes * self.greedyCost)

		self.bestPath = None
		self.bestPathCost = sys.maxint

		self.candidatesPerNode = g.numNodes / 2 

		# possible candidates for each city
		self.candidates = self.getCandidates(self.graph, self.candidatesPerNode)

		# scaling factors for visibility (inverse distance)
		self.beta = beta

		# probability of exploiting (1 - probability of exploring)
		self.exploitProb = exploitProb

		# the amount by which pheromone strengths should decay over time
		self.decay = decay

		# timing info
		self.totalTime = 0

	def solve(self):
		totalStartTime = time.time()
		
		for t in range(self.iterations):
			# Keep track of whether or not any ant finds a better solution than the best
			# so far, during this iteration.
			foundNewBestPath = False

			# maintain list of points from which ants have started a tour
			availableStartingPoints = set([i for i in range(self.numAnts)])

			# get current edge desirabilities for this iteration
			edgeDesirabilities = [[self.computeEdgeDesirability(row, col, self.pherStrengths) for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]

			# each ant is assigned a starting point and finds a tour from that starting point,
			# updating pheromone strength according to optimality of that path
			for k in range(self.numAnts):
				
				# replenish set of starting points if necessary (if numAnts > numNodes)
				if not availableStartingPoints:
					availableStartingPoints = set([i for i in range(self.numAnts)])

				# choose random starting point
				startingPoint = random.sample(availableStartingPoints, 1)[0]
				availableStartingPoints.remove(startingPoint)

				# each ant works off its own copy of edgeDesirabilities, so that
				# ants move independently at each time iteration
				path = self.findPath(copy.copy(edgeDesirabilities), startingPoint)
				# update bestPath if a new global best solution has been found.
				cost = path.cost()
				if cost < self.bestPathCost:
					self.bestPath = path
					self.bestPathCost = cost
					foundNewBestPath = True

				# each ant diminishes the pheromone strengths of the tour it has gone on,
				# to promote exploration by subsequent ants in this time iteration.
				self.weakUpdatePheromoneStrengths(path)

			if foundNewBestPath:
				# if some ant has found the best solution so far, only that
				# and should update the pheromone strengths along that path
				self.strongUpdatePheromoneStrengths(self.bestPath)

		totalEndTime = time.time()
		self.totalTime = (totalEndTime - totalStartTime) * 1000
		return self.bestPath

	# increase pheromone strengths along best path found so far
	def strongUpdatePheromoneStrengths(self, path):
		for row in range(len(self.pherStrengths)):
			for col in range(len(self.pherStrengths)):
				# if the edge row->col is not in the path, don't update the pheromone strength.
				# because path[i] gives the node connected to node i, if
				# path[row] == col, there is an edge from row to col.
				delta = 1 / path.cost() if path.path[row] == col else 0
				self.pherStrengths[row][col] = (1 - self.decay) * self.pherStrengths[row][col] + delta

	# diminishes pheromone strength along visited paths to encourage exploration
	def weakUpdatePheromoneStrengths(self, path):
		for row in range(len(self.pherStrengths)):
			for col in range(len(self.pherStrengths)):
				self.pherStrengths[row][col] = (1 - self.decay) * self.pherStrengths[row][col] + self.decay * self.initPherStrength

	# finds a path based on current pheromone strengths
	def findPath(self, edgeDesirabilities, startingPoint):
		p = Path(self.graph)
		length = 0
		current = startingPoint

		edgeDesirabilities = [[0 if col == current else edgeDesirabilities[row][col] for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]

		visited = set([current])

		while length < p.numNodes - 1:
			# determine whether to select a neighbor by exploring or exploiting
			neighbor = None
			
			# with probability self.exploitProb, exploit by just taking the most desirable unvisited node.
			if random.random() < self.exploitProb:
				neighbors = [i for i in range(self.graph.numNodes) if i not in visited]
				neighbor = max(neighbors, key=lambda n: edgeDesirabilities[current][n])

			# otherwise, explore like we did in ant system, first only choosing from candidates,
			# but selecting from all unvisited nodes if there are no unvisited candidates.
			# candidates are the self.candidatesPerNode closest neighbors to the current node.
			else:
				probs  = []
				unvisited = []
				candidates = []
				for i, prob in enumerate(edgeDesirabilities[current]):
					if i not in visited:
						unvisited.append(i)
						if self.graph.matrix[current][i] in self.candidates[current]:
							candidates.append(i)
							probs.append(prob) 

				# Normalize each row in edgeDesirabilities, as row i
				# represents the probabilities with which the ant should move
				# from city i to every other potential city.
				probs = self.normalizeRow(probs)

				if candidates:
					neighbor = np.random.choice(candidates, 1, p=probs)[0]

				# if there are no unvisited candidates, choose closest unvisited node
				else:
					neighbor = max(unvisited, key=lambda n: self.graph.matrix[current][n])

			visited.add(neighbor)
			p.addStep(current, neighbor)

			# make sure that this neighbor can never be visited again
			# by setting the probability that any city visits it to be 0.
			edgeDesirabilities = [[0 if col == neighbor else edgeDesirabilities[row][col] for col in range(self.graph.numNodes)] for row in range(self.graph.numNodes)]

			current = neighbor
			length += 1

		# add the final edge back to the starting point
		p.addStep(current, startingPoint)

		return p

	def computeEdgeDesirability(self, row, col, pherStrengths):
		# an edge's desirability is the product of the pheromone strength and the weight
		# of the edge (where weight is scaled by beta)

		# the desirability of a self-loop is 0.
		if row == col:
			return 0.0
		return pherStrengths[row][col] * ((1 / self.graph.matrix[row][col]) ** self.beta)

	def normalizeRow(self, row):
		rowSum = sum(row)
		row = [i / rowSum for i in row]
		return row
