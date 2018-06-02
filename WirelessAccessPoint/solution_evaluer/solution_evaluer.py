import math

from matplotlib.pyplot import plot, show
from matplotlib import pyplot as plt

class SolutionEvaluer:

    def __init__(self, prob):
        self._n_ap = prob.dim
        self._radius = prob.radius
        self._clients = prob.clients
        self._LB = -500
        self._UB = 500

    def _prepare_solution(self, sol):
        aps = []
        for index in range(0,len(sol),2):
            x,y = sol[index],sol[index+1]
            if (x >= self._LB and x <= self._UB) and (y >= self._LB and y <= self._UB):
                aps.append((x,y))
        return aps

    def plot(self, sol):
        aps = self._prepare_solution(sol)
        covered = 0
        for c in self._clients:
            plot(c[0], c[1], 'or')
            found = False
            index = 0
            while not found and index < len(sol):
                dist = math.sqrt((c[0] - sol[index]) ** 2 + (c[1] - sol[index + 1]) ** 2)
                if dist <= self._radius:
                    covered += 1
                    found = True
                index += 2

        for ap in aps:
            x,y = ap[0],ap[1]
            circle = plt.Circle((x, y), self._radius, color='green', alpha=0.3)
            plt.gcf().gca().add_artist(circle)
        show(block=False)
        plt.pause(0.5)
        print("Used Access Point:\t"+str(len(aps)))
        print("APs:")
        for el in aps:
            print("( "+str(el[0])+", "+str(el[1])+" )")
        print("Used Access Point:\t" + str(len(aps)))
        print("Covered "+str(covered)+" on "+str(len(self._clients)))