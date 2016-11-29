from graph import *
from plotter import *

errorMessage = """
usage: python solve.py <problem_size> <solver> <iterations>

solver options: AS (Ant System), ASE (Ant System with Elitist Ants), ACS (Ant Colony System)
problem options: small, medium, large, xlarge
iterations: must be an integer >= 1
         """

# quits program
def close():
   print errorMessage
   sys.exit(2)

# switch statement for problem size (size must be a valid option)
def getSize(size):
   return {
      'small': [Graph('test_data/small_15.txt'), 15],
      'medium': [Graph('test_data/western_sahara_29.txt'), 29],
      'large': [Graph('test_data/state_capitals_48.txt'), 48],
      'xlarge': [Graph('test_data/xlarge_131.txt'), 131],
   }.get(size)

# check if string is int
def isInt(n):
    try: 
        int(n)
        return True
    except ValueError:
        return False

def printPath(path):
  current = 0
  print current,
  for i in range(len(path)):
    print path[current],
    current = path[current]

# executes solver
def execute(s, graph):
   plotter = Plotter()
   path = s.solve()
   print '\nSolution cost:', format(path.length, '.2f')
   print 'Time (s):', format(s.totalTime / 1000, '.2f'), '\n'
   # print 'Path:', printPath(path.path)
   plotter.draw(graph.points, path.path)