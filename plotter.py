import matplotlib.pyplot as plt
import sys

class Plotter:
	# graph is a list of (x, y) coordinates
	# path is a list of indices into graph
	def draw(self, graph, path):
		maxX = -sys.maxint + 1
		minX = sys.maxint
		maxY = -sys.maxint + 1
		minY = sys.maxint

		# Plot all points
		isStart = True
		for x, y in graph:
			if isStart:
				plt.plot(x, y, 'go')
				isStart = False
			else:
				plt.plot(x, y, 'ro')

		# Plot path
		data = []
		
		for i in range(len(path)):
			if path[i] >= 0:
				x1, y1 = graph[i]
				x2, y2 = graph[path[i]]
				data += [(x1, x2), (y1, y2), 'b']
				if min(x1, x2) < minX:
					minX = min(x1, x2)
				if max(x1, x2) > maxX:
					maxX = max(x1, x2)
				if min(y1, y2) < minY:
					minY = min(y1, y2)
				if max(y1, y2) > maxY:
					maxY = max(y1, y2)

		buf = 10
		plt.axis([minX - buf, maxX + buf, minY - buf, maxY + buf])
		plt.plot(*data)
		plt.show()