from pygmo import *
from matplotlib import pyplot as plt
import math

class my_problem:
    def __init__(self, dim=10):
        self.dim = dim
    #
    #
    # def fitness(self, x):
    #     """The fitness() method is expected to return the fitness of the input decision vector """
    #     return [sum(x * x)]

    def fitness(self, x):
        obj = 0
        for i in range(3):
            obj += (x[2 * i - 2] - 3) ** 2 / 1000. - (x[2 * i - 2] - x[2 * i - 1]) + math.exp(20. * (x[2 * i - 2] - x[2 * i - 1]))
        ce1 = 4 * (x[0] - x[1]) ** 2 + x[1] - x[2] ** 2 + x[2] - x[3] ** 2
        ce2 = 8 * x[1] * (x[1] ** 2 - x[0]) - 2 * (1 - x[1]) + 4 * (x[1] - x[2]) ** 2 + x[0] ** 2 + x[2] - x[3] ** 2 + x[3] - x[4] ** 2
        ce3 = 8 * x[2] * (x[2] ** 2 - x[1]) - 2 * (1 - x[2]) + 4 * (x[2] - x[3]) ** 2 + x[1] ** 2 - x[0] + x[3] - x[4] ** 2 + x[0] ** 2 + x[4] - x[5] ** 2
        ce4 = 8 * x[3] * (x[3] ** 2 - x[2]) - 2 * (1 - x[3]) + 4 * (x[3] - x[4]) ** 2 + x[2] ** 2 - x[1] + x[4] - x[5] ** 2 + x[1] ** 2 + x[5] - x[0]
        ci1 = 8 * x[4] * (x[4] ** 2 - x[3]) - 2 * (1 - x[4]) + 4 * (x[4] - x[5]) ** 2 + x[3] ** 2 - x[2] + x[5] + x[2] ** 2 - x[1]
        ci2 = -(8 * x[5] * (x[5] ** 2 - x[4]) - 2 * (1 - x[5]) + x[4] ** 2 - x[3] + x[3] ** 2 - x[4])
        return [obj, ce1, ce2, ce3, ce4, ci1, ci2]

    def get_bounds(self):
        return ([-5] * 6, [5] * 6)

    def get_nic(self):
        return 6



    def get_nix(self):
        return 2

    # def get_bounds(self):
    #     """ get_bounds() is expected to return the box bounds of the problem, (lb,ub), which also implicitly define the dimension of the problem"""
    #     return ([-1] * self.dim, [1] * self.dim)


def test():
    prob = problem(my_problem(3))
    print(prob)
    #class pygmo.de(gen = 1, F = 0.8, CR = 0.9, variant = 2, ftol = 1e-6, xtol = 1e-6, seed = random)
    gen = 50
    F = 0.8
    CR = 0.9
    seed = 7
    algo = algorithm(pg.de(gen,F,CR))
    pop = population(prob, 10)
    pop = algo.evolve(pop)
    print(pop.champion_f)

def test2():
    udp = zdt(prob_id=1)
    pop = population(prob=udp, size=10, seed=3453412)
    ndf, dl, dc, ndl = fast_non_dominated_sorting(pop.get_f())
    pop = population(udp, 100)
    ax = plot_non_dominated_fronts(pop.get_f())
    plt.ylim([0, 6])
    plt.title("ZDT1: random initial population")
    algo = algorithm(moead(gen=250))
    pop = algo.evolve(pop)
    ax = plot_non_dominated_fronts(pop.get_f())
    plt.title("ZDT1: ... and the evolved population")

if __name__ == '__main__':
    prob = problem(my_problem(3))
    print(prob)