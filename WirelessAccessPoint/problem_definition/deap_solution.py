import math
import random

from deap import base
from deap import creator
from deap import tools
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
########################FITNESSS FUNCTION###############################################################################
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
    ret =  _coperture(to_eval), _AP_costs(to_eval, ap_graph),
    return ret
########################FUNZIONI DI APPOGGIO PER LA FITNESS#############################################################
def _AP_costs(individual, g):
    apc = len(individual) * AP_COST
    return apc

def wire_costs(individual, g):
    to_visit = [SOURCE_CABLE]
    cost = 0
    # BFS sul grafo per calcolare il costo del cavo
    while to_visit:
        v = to_visit.pop()
        if v == SOURCE_CABLE:
            source = {X: SOURCE_X, Y: SOURCE_Y}
        else:
            source = v
        for n in g[v]:
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

creator.create("Fitness", base.Fitness, weights=(1.0,-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

#Attribute Generator
toolbox.register("attr_AP", rand_ap)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_AP, N_AP)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the goal / fitness function
toolbox.register("evaluate", eval_fitness_costs_coperture)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", mutate_individual, mu=0.0, sigma=0.2, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)



# ----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=POP_SIZE)

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 1 and g < 5000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        #############MUTATION PHASE ####################################################################################
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    eval = SolutionEvaluer()
    eval.plot(best_ind)

if __name__ == "__main__":
    main()