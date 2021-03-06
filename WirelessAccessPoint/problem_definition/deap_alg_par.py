import networkx as nx
######COSTANTI DI AMBIENTE
SOL_TYPE = 3
X = 'x'
Y = 'y'
WIRE = 'wire'
AP_TYPE = 'type'
#PATH_CLINETS = "../200clients.txt"
PATH_CLINETS = "../400clients.txt"
SAVE_PATH = "./res400/"
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
P1 = 1000.0
P2 = 2.25*P1
P = [P1, P2]
if SOL_TYPE == 0:
    WEIGHTS = (1.0,-1.0, -1.0)
elif SOL_TYPE == 1:
    WEIGHTS = (1.0,-1.0)
elif SOL_TYPE == 2:
    WEIGHTS = (1.0, -1.0, 1.0)
elif SOL_TYPE == 3:
    WEIGHTS = (1.0, -1.0, -1.0, 1.0)
#WEIGHTS = (1.0,-1.0)
####################PARAMETRI ALGORITMO GENETICO########################################################################
N_IT = 1
N_GEN = 1400
POP_SIZE = 350
MIGRATION_INTERVAL = 100
MIGRATION_PERC = 2
N_MIGRATION = int((POP_SIZE/100)*MIGRATION_PERC) if int((POP_SIZE/100)*MIGRATION_PERC) > 1 else 1
#N_MIGRATION = 1
#STOP_CONDITION = [1]
STOP_CONDITION = 1
INDPB = 0.07
MU = 0
SIGMA = 5
TOURNAMENT_SIZE = 3
N_ISLES = 4

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
        if (dest < 0 and dest > N_AP) or source == dest:
            #L'ap è spento o disconnesso
            ap_graph.add_node(source)
        else:
            ap_graph.add_edge(source, dest)
    return ap_graph

########################################################################################################################
CLIENTS = load_clients(PATH_CLINETS)
