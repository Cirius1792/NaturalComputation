import math

from matplotlib import pyplot as plt
from WirelessAccessPoint.problem_definition.deap_alg_par import *

class SolutionEvaluer:

    def __init__(self, path=None):
        self._n_ap = N_AP
        self._radius = RADIUS
        self._clients = CLIENTS
        self._LB = LOWER_BOUND_GRID
        self._UB = UPPER_BOUND_GRID
        self._path = path

    def _prepare_solution(self, sol):
        to_eval = []
        if SOL_TYPE == 0:
            ap_graph = build_ap_graph(sol)
            for index in range(len(sol)):
                if nx.has_path(ap_graph, SOURCE_CABLE, index):
                    to_eval.append(sol[index])
        else:
            for index in range(len(sol)):
                if sol[index][WIRE] == 1:
                    to_eval.append(sol[index])
        return to_eval

    def plot(self, sol, save=False):
        aps = self._prepare_solution(sol)
        covered = 0
        plt.plot(SOURCE_X, SOURCE_Y, 'bo')
        for c in self._clients:
            plt.plot(c[0], c[1], 'or')
            found = False
            index = 0
            while not found and index < len(aps):
                dist = math.sqrt((c[0] - aps[index][X]) ** 2 + (c[1] - aps[index][Y]) ** 2)
                if dist <= RADIUS[aps[index][AP_TYPE]]:
                    covered += 1
                    found = True
                index += 1

        apt = [0,0]
        for ap in aps:
            apt[ap[AP_TYPE]] += 1
            x,y = ap[X],ap[Y]
            dst = sol[ap[WIRE]] if ap[WIRE] != N_AP+1 else {X:SOURCE_X, Y:SOURCE_Y}
            plt.plot([x,dst[X]], [y, dst[Y]], lw=0.5, C='gray') if SOL_TYPE == 0 else 0
            circle = plt.Circle((x, y), RADIUS[ap[AP_TYPE]], color='green', alpha=0.3)
            plt.gcf().gca().add_artist(circle)
        plt.title("Covered "+str(covered)+" AP1: "+str(apt[0])+" AP2: "+str(apt[1]))
        plt.savefig(self._path) if save else 0
        plt.show(block=False)
        plt.pause(0.5)

        print("Used Access Point:\t" + str(len(aps)),end="\t")
        print("AP1 : "+str(apt[0])+"\tAP2 : "+str(apt[1]))
        print("Covered "+str(covered)+" on "+str(len(self._clients)))

    def load_solution(self, path):
        pop = []
        pars = [X, Y, AP_TYPE, WIRE]
        f = open(path, 'r')
        line = f.readline()
        #Scarto i commenti iniziali
        while line[0] == '#' or line[0] == '\n':
            line = f.readline()
        spl = line.split(':')
        if spl[0] == 'N_INDIVIDUALS':
            n_indiviulas = int(spl[1])
        else:
            print("malformed file, N_INDIVIDUAL expected")
        for i in range(n_indiviulas):
            line = f.readline() #Scarto IND:#
            line = f.readline() #Scarto il commento sulla fitness
            line = f.readline()
            spl = line.split(':')
            if spl[0] == 'AP':
                n_ap = int(spl[1])
            else:
                print("malformed file, AP expected")
            aps = []
            for j in range(n_ap):
                line = f.readline()
                spl = line.split(',') #Divido i diversi parametri di un singolo gene
                ap = dict()
                for i in range(len(spl)):
                    s = spl[i].split(':')
                    ap[pars[i]] = float(s[1]) if i < 2 else int(s[1])
                aps.append(ap)
            pop.append(aps)
        return pop

if __name__ == '__main__':
    sol_eval = SolutionEvaluer()
    pop = sol_eval.load_solution("../problem_definition/res/1529580332.6880443/1529580332.6880443.txt")
    print(str(pop[0]))
    for el in pop:
        sol_eval.plot(el)