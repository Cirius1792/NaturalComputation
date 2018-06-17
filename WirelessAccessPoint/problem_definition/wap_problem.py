import math
import time

from WirelessAccessPoint.APPlotter import APPlotter
import numpy as np
import pygmo as pg

from WirelessAccessPoint.solution_evaluer.priorityq import PriorityQueue
from WirelessAccessPoint.solution_evaluer.solution_evaluer import SolutionEvaluer

SPACE_L = -500.00
SPACE_U = 500.00
POP_SIZE = 200

class wap_problem:

    def __init__(self, area, bounds, obj):
        """obj:
            -  'CLIENTS'
            -   'RADIUS'
            -   'COSTS'
        """
        self.costs = obj['COSTS']
        self.bounds = bounds
        self.clients = obj['CLIENTS']
        self.radius = obj['RADIUS']
        self.dim = math.ceil(area/(2*math.pi*(self.radius**2)))


    def fitness (self, x):
        """Voglio minimizzare il costo e massimizzare la copertura"""
        #ret = [sum([self._AP_cost(x), self._coperture(x)])]
        ret = [self._coperture(x)]
        return ret


    def get_bounds(self):
        return ([self.bounds['S_LB']-1]*self.dim*2, [self.bounds['S_UB']+1]*self.dim*2)

    def _AP_cost(self, x):
        n_ap = 0
        for el in x:
            #visto che lo spazio è quadrato posso considerare gli elementi singolarmente
            #ed allo stesso modo, dividendo poi per due n_ap
            if el <= self.bounds['S_UB'] and el >= self.bounds['S_LB']:
                n_ap += 1
        n_ap = math.floor(n_ap / 2)
        # for index in range(0, len(x), 2):
        #     x, y = x[index], x[index + 1]
        #     if (x >= self.bounds['S_LB'] and x <= self.bounds['S_UB']) and (y >= self.bounds['S_LB']and y <= self.bounds['S_UB']):
        #         n_ap += 1
                
        return n_ap/self.dim

    def _coperture(self, x):
        covered = 0
        for client in self.clients:
            found = False
            index = 0
            while not found and index <len(x):
                dist = math.sqrt( (client[0] - x[index])**2 + (client[1] - x[index+1])**2 )
                if dist <= self.radius:
                    covered += 1
                    found = True
                index += 2
        return -covered/len(self.clients)


def load_clients(path):
    file = open(path)
    clients = []
    for line in file:
        if line:
            coordinates = line.split()
            if len(coordinates) > 1:
                x,y = float(coordinates[0]), float(coordinates[1])
                clients.append((x,y))
    return clients

if __name__ == '__main__':
    path = "C:/Users/CiroLucio/PycharmProjects/NaturalComputation/WirelessAccessPoint/200clients.txt"
    clients = load_clients(path)
    #clinets = clients[0:math.ceil(len(clients)/2)]
    #plotter = APPlotter(clients)
    bounds = {'S_LB':SPACE_L, 'S_UB':SPACE_U}
    obj = {'CLIENTS':clients, 'RADIUS':50.0, 'COSTS':1.0}
    area = 1000*1000
    mp = wap_problem(area,bounds,obj)
    prob = pg.problem(mp)
    print(prob)
    sol_eval = SolutionEvaluer(mp)
    #t = [np.random.rand() for i in range(400)]
    # f = prob.fitness(t)
    # print(str(f))
    # print(prob)
    gen = 10
    F = 0.7
    CR = 0.85
    seed = 7
    algo = pg.algorithm(pg.de(gen,F,CR))
    #algo = pg.algorithm(pg.moead(gen=250))
    #algo = pg.algorithm(pg.sga(gen = gen))
    algo.set_verbosity(50)
    # # #SINGOLA EVOLUZIONE
    # pop = pg.population(prob, size=30, seed=seed)
    # start = time.time()
    # pop = algo.evolve(pop)
    # stop = time.time() - start
    # best = pop.get_x()[pop.best_idx()]
    # print("Champion's Fitness: \t"+str(pop.champion_f)+"\t time: "+str(stop))
    # sol_eval.plot(best)
    # #MULTICORE
    archi = pg.archipelago(4, algo=algo, prob=prob, pop_size=30)
    archi.evolve(4)
    archi.wait()
    res = [isl.get_population().champion_f for isl in archi]
    best = None
    for isl in archi:
        if best and best.champion_f < isl.get_population().champion_f:
            best = isl.get_population()
        else:
            best = isl.get_population()
    sol_eval.plot(best.get_x()[best.best_idx()])

def save_pop(isl):
    pop = isl.get_population()
    pq = PriorityQueue()
    for ind in pop:
        pq.add(ind)