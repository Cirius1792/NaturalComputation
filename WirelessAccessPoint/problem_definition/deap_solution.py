import datetime
import math
import time
import matplotlib.pyplot as plt
from statistics import stdev

import numpy
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


from joblib import Parallel, delayed
import os

import networkx as nx

import warnings
warnings.filterwarnings("ignore")

from WirelessAccessPoint.solution_evaluer.solution_evaluer2 import SolutionEvaluer
from WirelessAccessPoint.problem_definition.deap_alg_par import *
########################################################################################################################

######METODO PER LA CREAZIONE DI UN GENE ###############################################################################
def rand_ap():
    x = random.uniform(UPPER_BOUND_GRID, LOWER_BOUND_GRID)
    y = random.uniform(UPPER_BOUND_GRID, LOWER_BOUND_GRID)
    cable = random.randint(-1, N_AP+1)
    ap_type = random.randint(0,1)
    return {X:x, Y:y, WIRE:cable, AP_TYPE:ap_type}
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
    return _coperture(to_eval), _AP_costs(to_eval), wire_costs(individual,ap_graph)
    #return _coperture(to_eval), _AP_costs(to_eval)
######################  FUNZIONI DI APPOGGIO PER LA FITNESS  ###########################################################
def _AP_costs(individual):
    apc = 0
    for ap in individual:
        apc += AP_COST[ap[AP_TYPE]]
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
            if dist <= RADIUS[individual[index][AP_TYPE]]:
                covered += 1
                found = True
            index += 1
    return covered / len(clients)

def _signal_intensity(individual):
    clients = CLIENTS
    s_int = 0
    for client in clients:
        for ap in individual:
            dist = math.sqrt((client[0] - ap[X]) ** 2 + (client[1] - ap[Y]) ** 2)
            if dist < RADIUS[ap[AP_TYPE]]:
                s_int += P/(4*math.pi*(dist**2))
    return s_int/len(clients)
########################################################################################################################
##################### FUNZIONE CUSTOM DI MUTAZIONE  ####################################################################
def mutate_individual(individual, mu=0.0, sigma=1, indpb=INDPB):
    for i in range(len(individual)):
        #Mutate X
        if random.random() < indpb:
            individual[i][X] += random.gauss(mu, sigma)
        #Mutate Y
        if random.random() < indpb:
            individual[i][Y] += random.gauss(mu, sigma)
        #Mutate WIRE
        if random.random() < indpb:
            #individual[i][WIRE] = random.randint(-1, N_AP)
            if individual[i][WIRE] >= 0  and individual[i][WIRE] <= N_AP-1:
                individual[i][WIRE] = individual[individual[i][WIRE]][WIRE]
            else:
                individual[i][WIRE] = random.randint(-1, N_AP + 1)
        #Mutate AP_TYPE
        if random.random() < indpb:
            individual[i][AP_TYPE] = abs(1 - individual[i][AP_TYPE])
    return individual,
########################################################################################################################
###################### STOP CONDITION ##################################################################################
def stop_cond(islands,STOP_CONDITION = 0):
    if STOP_CONDITION != 0:
        cur_ind = [[] for el in WEIGHTS]
        for island in islands:
            best_ind = tools.selBest(island, 1)[0]
            for i in range(len(WEIGHTS)):
                cur_ind[i].append(best_ind.fitness.values[i])
        #print("dev0: "+"{0:.3f}".format(stdev(cur_ind[0]))+"\tdev1: "+"{0:.3f}".format(stdev(cur_ind[1]))+"\tdev2: "+"{0:.3f}".format(stdev(cur_ind[2])))
        for i in range(len(WEIGHTS)):
            print("dev"+str(i)+":\t"+"{0:.3f}".format(stdev(cur_ind[i]))+"\t", end="\t")
        print("")
        if stdev(cur_ind[0]) < 0.05 and stdev(cur_ind[1]) < 10 and stdev(cur_ind[2]) < 100:
            return True

    return False
########################################################################################################################
###################### OUTPUT FUNCTIONS#################################################################################

def plot_stats(logbook):
    min_labels = ["min coperture", "min ap cost", "min wire cost"]
    max_labels = ["max coperture", "max ap cost", "max wire cost"]
    min_col= ["c-", "b-"]
    max_col= ["k-", "r-"]
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_max = logbook.select("max")

    fig, ax1 = plt.subplots()
    #COPERTURE
    # to_plot = [fit_mins[j][0] for j in range(len(fit_mins))]
    # line0 = ax1.plot(gen, to_plot, min_col[0], label=min_labels[0])


    to_plot = [fit_max[j][0] for j in range(len(fit_mins))]
    line3 = ax1.plot(gen, to_plot, max_col[1], label=max_labels[0])

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Coperture", color="r")
    for tl in ax1.get_yticklabels():
        tl.set_color("r")

    #AP_COSTS
    ax2 = ax1.twinx()
    to_plot = [fit_mins[j][1] for j in range(len(fit_mins))]
    line1 = ax2.plot(gen, to_plot, min_col[1], label=min_labels[1])

    to_plot = [fit_mins[j][2] for j in range(len(fit_mins))]
    line0 = ax2.plot(gen, to_plot, '-c', label=min_labels[2])

    # to_plot = [fit_max[j][1] for j in range(len(fit_mins))]
    # line2 = ax2.plot(gen, to_plot, max_col[0], label=max_labels[1])
    ax2.set_ylabel("Cost", color="b")
    for tl in ax2.get_yticklabels():
        tl.set_color("b")

    lns = line0 + line1 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    plt.show()
    return fig


def print_output(pop, it=None, n_ind=1):
    print("############################### FINAL STATS ###################################")
    if it is None:
        it = N_GEN
    best_inds = tools.selBest(pop, n_ind)
    for best_ind in best_inds:
        print("Best individual is:\n\t %s\n\t %s" % (best_ind, best_ind.fitness.values))
        eval = SolutionEvaluer()
        eval.plot(best_ind)
        print("")

def print_infos():
    print("############################### SETTINGS ###################################")
    print("Genetic Algorithm Parameters:")
    print("Max gen: \t\t"+str(N_GEN))
    print("Migrations Interval: \t"+str(MIGRATION_INTERVAL)+"\twith "+str(MIGRATION_PERC)+"% migrating individual")
    print("N Islands:\t" + str(N_ISLES)+" with pop size:\t"+str(POP_SIZE))
    print("INDPB:\t"+str(INDPB))
    print("Tournament size:\t "+str(TOURNAMENT_SIZE))
    print("CXPB:\t%s\tMUTPB:\t%s" % (CXPB, MUTPB))
    print("#############################################################################")

def print_ind(ind):
    out = ""
    out += X+":"+str(ind[X])+", "
    out += Y+":"+str(ind[Y])+", "
    out += AP_TYPE+":"+str(ind[AP_TYPE])+", "
    out += WIRE+":"+str(ind[WIRE])+"\n"
    return out

def save_results(path, pop):
    run_id = time.time()
    if not os.path.exists(path+str(run_id)):
        os.makedirs(path+str(run_id))
    path = path+str(run_id)+"/"
    f = open(path+str(run_id)+".txt",'w')
    f.write("\n################################ SETTINGS ###################################")
    f.write("\n#RUN ID: "+str(run_id))
    f.write("\n#Genetic Algorithm Parameters:")
    f.write("\n#Max gen: \t\t"+str(N_GEN))
    f.write("\n#Migrations Interval: \t"+str(MIGRATION_INTERVAL)+"\twith "+str(MIGRATION_PERC)+"% migrating individual")
    f.write("\n#N Islands:\t" + str(N_ISLES)+" with pop size:\t"+str(POP_SIZE))
    f.write("\n#INDPB:\t"+str(INDPB))
    f.write("\n#Tournament size:\t "+str(TOURNAMENT_SIZE))
    f.write("\n#CXPB:\t%s\tMUTPB:\t%s" % (CXPB, MUTPB))
    f.write("\n##############################################################################")
    f.write("\nN_INDIVIDUALS:" + str(len(pop))+"\n")
    i = 0

    for individual in pop:
        eval = SolutionEvaluer(path=path+"run"+str(run_id)+"_ind"+str(i)+".png")
        f.write("IND:"+str(i)+"\n")
        i += 1
        f.write("#fitness: \t"+str(individual.fitness.values)+"\n")
        ap_graph = build_ap_graph(individual)
        to_eval = list(individual)
        #to_eval = []
        # for index in range(len(individual)):
        #     if nx.has_path(ap_graph, SOURCE_CABLE, index):
        #         to_eval.append(individual[index])
        f.write("AP:"+str(len(to_eval))+"\n")
        for ap in to_eval:
            f.write(print_ind(ap))
        eval.plot(individual, save=True)
########################################################################################################################

creator.create("Fitness", base.Fitness, weights=WEIGHTS)
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

#Attribute Generator
toolbox.register("attr_AP", rand_ap)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_AP, N_AP)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_AP, N_AP)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the goal / fitness function
toolbox.register("evaluate", eval_fitness_costs_coperture)

# register the crossover operator
toolbox.register("mate", tools.cxUniform, indpb=INDPB)

toolbox.register("mutate", mutate_individual, mu=MU, sigma=SIGMA, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)



# ----------


def single_evolver(pop=None, n_gen=N_GEN, hof=None, verbose=True):

    if pop is None:
        random.seed(64)
        pop = toolbox.population(n=POP_SIZE)
    if hof is None:
        hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=n_gen,
                                   stats=stats, halloffame=hof, verbose=False)
    if verbose:
        best_inds = tools.selBest(hof, 1)
        for best_ind in best_inds:
            print("Best individual fitness: \t"+str(best_ind.fitness.values))
    return pop, log, hof


def parallel_evolution():
    print_infos()
    islands = [toolbox.population(n=POP_SIZE) for i in range(N_ISLES)]
    with Parallel(n_jobs=N_JOBS) as parallel:
        hof = tools.ParetoFront()
        it = 0
        while it == 0 or (it < N_GEN and not stop_cond(islands,STOP_CONDITION)):
        #for i in range(0, generations, migration_interval):
            print("\nIteration: "+str(it))
            res = parallel(delayed(single_evolver)(pop=island, n_gen=MIGRATION_INTERVAL, hof=hof, verbose=True) for island in islands)
            islands = []
            for pop, logbook, hofi in res:
                hof.update(pop)
                islands.append(pop)
            tools.migRing(islands, N_MIGRATION, tools.selBest) if N_ISLES > 1 else 0
            it += MIGRATION_INTERVAL
    #for island in islands:
    #    print_output(island, it)
    #print_output(hof, it,n_ind=N_ISLES)
    return hof

def multi_islands():
    random.seed(64)
    NISLES = 6
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

def parallel_main():
    random.seed(64)
    best_ind=tools.HallOfFame(int(POP_SIZE))
    start = time.time()
    for i in range(N_IT):
        print("Iteration: "+str(i))
        pop = parallel_evolution()
        best_ind.update(pop)
    stop = time.time()-start
    print_output(best_ind,n_ind=5)
    save_results(SAVE_PATH,  tools.selBest(best_ind, 10)) if SAVE_DATA else 0
    print("Time: \t "+"{0:.4f}".format(stop))

def single_main():
    start = time.time()
    pop,log,hof = single_evolver(verbose=False)
    fig = plot_stats(log)
    stop = time.time()-start
    #print_output(hof, n_ind=5)
    print("Time: \t "+"{0:.4f}".format(stop))


if __name__ == "__main__":
    #single_main()
    parallel_main()