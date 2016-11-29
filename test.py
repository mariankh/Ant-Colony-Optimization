from graph import *
from plotter import *

graph = Graph('grid48_xy.txt')
plotter = Plotter()
numAnts = 48
numIterations = 10
# solver = AntSystem(graph, numAnts, numIterations)
solver = AntColonySystem(graph, numAnts, numIterations)
# solver = Greedy(graph)
path = solver.solve()

print 'path:', path.path
print 'cost:', path.length
print 'total time:', solver.totalTime

print 'time (s):', format(solver.totalTime / 1000, '.2f')
print format(path.length, '.2f')
plotter.draw(graph.points, path.path)

# 15 nodes: 284
# 38 nodes: 6656
# 131 nodes: 564