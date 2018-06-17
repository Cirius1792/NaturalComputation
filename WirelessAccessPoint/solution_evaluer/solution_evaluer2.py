import math

from matplotlib.pyplot import plot, show
from matplotlib import pyplot as plt
from WirelessAccessPoint.problem_definition.deap_alg_par import *

class SolutionEvaluer:

    def __init__(self):
        self._n_ap = N_AP
        self._radius = RADIUS
        self._clients = CLIENTS
        self._LB = LOWER_BOUND_GRID
        self._UB = UPPER_BOUND_GRID

    def _prepare_solution(self, sol):
        ap_graph = build_ap_graph(sol)
        to_eval=[]
        for index in range(len(sol)):
            if nx.has_path(ap_graph, SOURCE_CABLE, index):
                to_eval.append(sol[index])
        return to_eval

    def plot(self, sol):
        aps = self._prepare_solution(sol)
        covered = 0
        for c in self._clients:
            plot(c[0], c[1], 'or')
            found = False
            index = 0
            while not found and index < len(sol):
                dist = math.sqrt((c[0] - sol[index][X]) ** 2 + (c[1] - sol[index][Y]) ** 2)
                if dist <= self._radius:
                    covered += 1
                    found = True
                index += 1

        for ap in aps:
            x,y = ap[X],ap[Y]
            circle = plt.Circle((x, y), self._radius, color='green', alpha=0.3)
            plt.gcf().gca().add_artist(circle)
        show(block=False)
        plt.pause(0.5)
        print("Used Access Point:\t" + str(len(aps)))
        print("Covered "+str(covered)+" on "+str(len(self._clients)))