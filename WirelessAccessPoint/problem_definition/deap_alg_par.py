import networkx as nx
######COSTANTI DI AMBIENTE
SOL_TYPE = 1
X = 'x'
Y = 'y'
WIRE = 'wire'
AP_TYPE = 'type'
PATH_CLINETS = "../200clients.txt"
#PATH_CLINETS = "../400clients.txt"
SAVE_PATH = "./res/"
SAVE_DATA = True
PLOT_PATH = False
N_JOBS = 4
#Costanti usate per modellare il problema:

N_AP = 100
SOURCE_CABLE = N_AP+1
UPPER_BOUND_GRID = 500.0
LOWER_BOUND_GRID = -500.0
RADIUS = [50, 75]
SOURCE_X = -250.0
SOURCE_Y = 250.0
AP_COST = [10, 15]
WIRE_COST = 1
P = 100.0
WEIGHTS = (1.0,-1.0, -1.0)if SOL_TYPE == 0 else (1.0,-1.0)
#WEIGHTS = (1.0,-1.0)
####################PARAMETRI ALGORITMO GENETICO########################################################################
N_IT = 1
N_GEN = 200
POP_SIZE = 300
MIGRATION_INTERVAL = 300
MIGRATION_PERC = 1
#N_MIGRATION = int((POP_SIZE/100)*MIGRATION_PERC) if int((POP_SIZE/100)*MIGRATION_PERC) > 1 else 1
N_MIGRATION = 1
#STOP_CONDITION = [1]
STOP_CONDITION = 1
INDPB = 0.2
MU = 0
SIGMA = 5
TOURNAMENT_SIZE = 2
N_ISLES = 1

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.25

######################UTILITY###########################################################################################
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

def build_ap_graph(individual):
    ap_graph = nx.Graph()
    ap_graph.add_node(SOURCE_CABLE)
    for index in range(len(individual)):
        source = index
        dest = individual[index][WIRE]
        if dest == -1 or source == dest:
            #L'ap è spento o disconnesso
            ap_graph.add_node(source)
        else:
            ap_graph.add_edge(source, dest)
    return ap_graph

########################################################################################################################
CLIENTS = load_clients(PATH_CLINETS)
