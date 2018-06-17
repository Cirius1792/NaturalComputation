import math
import time
from multiprocessing import freeze_support

import numpy
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from multiprocessing import Event, Pipe, Process


from joblib import Parallel, delayed
import networkx as nx

from WirelessAccessPoint.solution_evaluer.solution_evaluer2 import SolutionEvaluer
from WirelessAccessPoint.problem_definition.deap_alg_par import *
########################################################################################################################
######METODO PER LA CREAZIONE DI UN GENE ###############################################################################
def rand_ap():
    x  = random.uniform(UPPER_BOUND_GRID, LOWER_BOUND_GRID)
    y  = random.uniform(UPPER_BOUND_GRID, LOWER_BOUND_GRID)
    cable  = random.randint(-1, N_AP+1)
    return {X:x,Y:y,WIRE:cable}
########################################################################################################################
######################  FITNESSS FUNCTION  ############################################################################
def eval_fitness_costs_coperture(individual):
    #Utilizzo un grafo per modellare l'interconnessione fra gli ap. Successivamente il grafo verrà utilizzato per ricercare
    #un percorso fra ogni ap e la Sorgente del segnale, se un path non esiste l'ap non viene considerato per la valutazione
    #della fitnes, risultando scollegato dalla rete
    ap_graph = build_ap_graph(individual)
    to_eval = []
    for index in range(len(individual)):
        if nx.has_path(ap_graph, SOURCE_CABLE, index):
            to_eval.append(individual[index])
    #ret =  _coperture(to_eval), _AP_costs(to_eval, ap_graph), wire_costs(to_eval, ap_graph),
    return _coperture(to_eval), _AP_costs(to_eval),wire_costs(individual,ap_graph)
######################  FUNZIONI DI APPOGGIO PER LA FITNESS  ###########################################################
def _AP_costs(individual):
    apc = len(individual) * AP_COST
    return apc

def wire_costs(individual, g):
    to_visit = [SOURCE_CABLE]
    cost = 0
    # BFS sul grafo per calcolare il costo del cavo
    visited = set()
    while to_visit:
        v = to_visit.pop()
        visited.add(v)
        if v == SOURCE_CABLE:
            source = {X: SOURCE_X, Y: SOURCE_Y}
        else:
            source = individual[v]
        for n in g[v]:
            if n not in visited:
                dist = math.sqrt((source[X] - individual[n][X]) ** 2 + (source[Y] - individual[n][Y]) ** 2)
                cost += dist * WIRE_COST
                to_visit.append(n)
    return cost

def _coperture(individual):
    clients = CLIENTS
    covered = 0
    for client in clients:
        found = False
        index = 0
        while not found and index < len(individual):
            dist = math.sqrt((client[0] - individual[index][X]) ** 2 + (client[1] - individual[index][Y]) ** 2)
            if dist <= RADIUS:
                covered += 1
                found = True
            index += 1
    return covered / len(clients)
########################################################################################################################
##################### FUNZIONE CUSTOM DI MUTAZIONE  ####################################################################
def mutate_individual(individual, mu=0.0, sigma=0.2, indpb=INDPB):
    for i in range(len(individual)):
        #Muto X
        if random.random() < indpb:
            individual[i][X] += random.gauss(mu, sigma)
        #Muto Y
        if random.random() < indpb:
            individual[i][Y] += random.gauss(mu, sigma)
        #Muto WIRE
        if random.random() < indpb:
            individual[i][WIRE] = random.randint(-1, N_AP+1)
    return individual,

########################################################################################################################

creator.create("Fitness", base.Fitness, weights=(1.0,-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
#per l'esecuzione multi processore
#pool = multiprocessing.Pool()
#toolbox.register("map", pool.map)

#Attribute Generator
toolbox.register("attr_AP", rand_ap)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_AP, N_AP)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_AP, N_AP)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the goal / fitness function
toolbox.register("evaluate", eval_fitness_costs_coperture)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", mutate_individual, mu=MU, sigma=SIGMA, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)



# ----------


def main2(pop=None, n_gen=N_GEN, verbose=True):
    random.seed(64)
    if pop is None:
        pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=n_gen,
                                   stats=stats, halloffame=hof, verbose=verbose)
    if verbose:
        best_inds = tools.selBest(hof, 5)
        for best_ind in best_inds:
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            eval = SolutionEvaluer()
            eval.plot(best_ind)
    return pop, log


def parallel_evolution():

    random.seed(64)

    NISLES = 4
    islands = [toolbox.population(n=300) for i in range(NISLES)]
    migration_interval = 5
    generations = 50
    with Parallel(n_jobs=4) as parallel:
        for i in range(0, generations, migration_interval):
            res = parallel(delayed(main2)(island,migration_interval, False) for island in islands)
            islands = [pop for pop, logbook in res]
            tools.migRing(islands, int(POP_SIZE/10), tools.selBest)

    for island in islands:
        best_inds = tools.selBest(island, 1)
        for best_ind in best_inds:
            print("Best individual is:\n\t %s\n\t %s" % (best_ind, best_ind.fitness.values))
            eval = SolutionEvaluer()
            eval.plot(best_ind)
            print("")

def multi_islands():
    random.seed(64)
    NISLES = 5
    islands = [toolbox.population(n=300) for i in range(NISLES)]
    # Unregister unpicklable methods before sending the toolbox.
    toolbox.unregister("attr_AP")
    toolbox.unregister("individual")
    toolbox.unregister("population")

    NGEN, FREQ = 50, 5
    toolbox.register("algorithm", algorithms.eaSimple, toolbox=toolbox,
                     cxpb=0.5, mutpb=0.2, ngen=FREQ, verbose=False)
    for i in range(0, NGEN, FREQ):
        results = toolbox.map(toolbox.algorithm, islands)
        islands = [pop for pop, logbook in results]
        tools.migRing(islands, 15, tools.selBest)

    for island in islands:
        best_inds = tools.selBest(island, 1)
        for best_ind in best_inds:
            print("Best individual is:\n\t %s\n\t %s" % (best_ind, best_ind.fitness.values))
            eval = SolutionEvaluer()
            eval.plot(best_ind)
            print("")
    return islands

if __name__ == "__main__":
    start = time.time()
    #multi_islands()
    parallel_evolution()
    stop = time.time()-start
    print("Time: \t "+"{0:.4f}".format(stop))