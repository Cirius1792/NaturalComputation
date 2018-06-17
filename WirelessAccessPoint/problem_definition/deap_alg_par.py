import networkx as nx
######COSTANTI DI AMBIENTE
X = 'x'
Y = 'y'
WIRE = 'wire'
PATH_CLINETS = "C:/Users/CiroLucio/PycharmProjects/NaturalComputation/WirelessAccessPoint/200clients.txt"
#Costanti usate per modellare il problema:
AP_TYPE = 1

N_AP = 10
SOURCE_CABLE = N_AP+1
UPPER_BOUND_GRID = 500.0
LOWER_BOUND_GRID = -500.0
RADIUS = 50 if AP_TYPE == 1 else 75
SOURCE_X = -250.0
SOURCE_Y =250.0
AP_COST = 10 if AP_TYPE == 1 else 15
WIRE_COST = 1
####################PARAMETRI ALGORITMO GENETICO########################################################################
N_GEN = 200
INDPB = 0.07
MU = 0.5
SIGMA = 1
TOURNAMENT_SIZE = 3
POP_SIZE = 200
# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

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
