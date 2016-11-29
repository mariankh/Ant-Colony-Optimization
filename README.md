# Ant Colony Optimization

## Contributors
Billy Cox, Erin Masatsugu

## Introduction
The goal of our project was to construct a polynomial time algorithm to find good approximate solutions to instances of the Traveling Salesman Problem (TSP). The Traveling Salesman Problem poses the following question: given a graph of `n` nodes, what is the shortest (least costly) tour through the graph? A tour is defined as a path which starts from some point `s`, visits each node in the graph exactly once, and then returns back to `s` at the end. This is an NP-Hard problem.

The intuition behind the Ant Colony Optimization algorithm comes from the way ant colonies forage for food. An ant's goal is to find food and bring it back to the nest. As an ant crawls, it leaves pheromones behind itself, which other ants can detect. This pheromone evaporates, so the strength of the scent decreases over time. When an ant is looking for food, it initially wanders randomly, but when it detects a pheromone trail, it follows its nose with a certain probability. Because the pheromone scent strength decays over time, shorter paths which can be traversed more frequently accumulate stronger pheromone scents. Eventually, the ants collectively find the shortest path between their nest and the food. This process gives a model for a natural reinforcement learning algorithm.

For implementation details, check out `Report.pdf` or `Poster.pdf`.

## Usage

To try it out, type

`python solve.py [problem size] [solver] [iterations]`

Where `problem size` is one of `small`, `medium`, `large`, or `xlarge`, `solver` is one of the values `AS`, `ASE`, or `ACS` (different algorithms, explained in `Report.pdf`), and `iterations` is the desired number of iterations.
