import math
from WirelessAccessPoint.APPlotter import APPlotter
import numpy as np
import pygmo as pg

SPACE_L = -500.00
SPACE_U = 500.00
POP_SIZE = 200

class wap_problem:

    def __init__(self, bounds, obj):
        """obj:
            -  'CLIENTS'
            -   'RADIUS'
            -   'COSTS'
        """
        self.dim = len(obj['CLIENTS'])
        self.costs = obj['COSTS']
        self.bounds = bounds
        self.clients = obj['CLIENTS']
        self.radius = obj['RADIUS']

    def fitness (self, x):
        """Voglio minimizzare il costo e massimizzare la copertura"""
        ret = [self._AP_cost(x), self._coperture(x)]

        return [self._coperture(x)]

    def get_bounds(self):

        return ([self.bounds['S_LP']-1]*self.dim*2, [self.bounds['S_UP']+1]*self.dim*2)

    def _AP_cost(self, x):
        n_ap = 0
        for el in x:
            #visto che lo spazio è quadrato posso considerare gli elementi singolarmente
            #ed allo stesso modo, dividendo poi per due n_ap
            if el <= self.bounds['S_UP'] and el >= self.bounds['S_LP']:
                n_ap += 1
        n_ap = n_ap / 2
        #return  n_ap*self.costs['AP']
        return n_ap/(len(self.clients))

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
        return covered/len(self.clients)


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
    plotter = APPlotter(clients)
    bounds = {'S_LP':SPACE_L, 'S_UP':SPACE_U}
    obj = {'CLIENTS':clients, 'RADIUS':50.0, 'COSTS':1.0}
    prob = pg.problem(wap_problem(bounds,obj))
    print(prob)
    #t = [np.random.rand() for i in range(400)]
    # f = prob.fitness(t)
    # print(str(f))
    # print(prob)
    gen = 100
    F = 0.8
    CR = 0.9
    seed = 7
    algo = pg.algorithm(pg.de(gen,F,CR))
    #SINGOLA EVOLUZIONE
    # pop = pg.population(prob, 30)
    # pop = algo.evolve(pop)
    # best = pop.get_x()[pop.best_idx()]
    #print(best)
    archi = pg.archipelago(4, algo=algo, prob=prob, pop_size=20)
    archi.evolve(5)
    archi.wait()
    res = [isl.get_population().champion_f for isl in archi]
    print(res)
    #plotter.update(best, obj['RADIUS'])