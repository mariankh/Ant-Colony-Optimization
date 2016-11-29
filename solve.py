import sys
from graph import *
from plotter import *
from solve_helpers import *

if len(sys.argv) != 4:
   close()

problem, solver, iterations = sys.argv[1].lower(), sys.argv[2].lower(), sys.argv[3]
if problem not in ['small', 'medium', 'large', 'xlarge'] or solver not in ['as','ase','acs'] or not isInt(iterations):
   close()

size = getSize(problem)

if solver == 'as':
   s = AntSystem(*(size + [int(iterations)]))
   execute(s, size[0])

elif solver == 'ase':
   s = AntSystem(*(size + [int(iterations)]), elitist=True)
   execute(s, size[0])

elif solver == 'acs':
   s = AntColonySystem(*(size + [int(iterations)]))
   execute(s, size[0])

