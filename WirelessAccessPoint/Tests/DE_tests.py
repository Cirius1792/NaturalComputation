import math
import time

from WirelessAccessPoint.problem_definition.wap_problem import *
import numpy as np
import pygmo as pg
from WirelessAccessPoint.solution_evaluer.solution_evaluer import SolutionEvaluer
import pickle



if __name__ == '__main__':

    path = "C:/Users/CiroLucio/PycharmProjects/NaturalComputation/WirelessAccessPoint/200clients.txt"
    clients = load_clients(path)
    # clinets = clients[0:math.ceil(len(clients)/2)]
    # plotter = APPlotter(clients)
    bounds = {'S_LB': SPACE_L, 'S_UB': SPACE_U}
    obj = {'CLIENTS': clients, 'RADIUS': 50.0, 'COSTS': 1.0}
    area = 1000 * 1000
    mp = wap_problem(area, bounds, obj)
    prob = pg.problem(mp)
    print(prob)
    sol_eval = SolutionEvaluer(mp)
    # t = [np.random.rand() for i in range(400)]
    # f = prob.fitness(t)
    # print(str(f))
    # print(prob)
    gen = 500
    F = 0.7
    CR = 0.7
    seed = 7
    res = []
    for f in range(F,0.9,0.05):
        for cr in range(CR,0.9,0.05):
            algo = pg.algorithm(pg.de(gen, f, CR))
            algo.set_verbosity(50)
            # #MULTICORE
            archi = pg.archipelago(4, algo=algo, prob=prob, pop_size=30)
            archi.evolve(5)
            archi.wait()
            res = [isl.get_population().champion_f for isl in archi]
            best = None
            for isl in archi:
                if best and best.champion_f < isl.get_population().champion_f:
                    best = isl.get_population()
                else:
                    best = isl.get_population()
            sol_eval.plot(best.get_x()[best.best_idx()])