import math

def _AP_cost(x):
    n_ap = 0
    for el in x:
        # visto che lo spazio è quadrato posso considerare gli elementi singolarmente
        # ed allo stesso modo, dividendo poi per due n_ap
        if el <= self.bounds['S_UB'] and el >= self.bounds['S_LB']:
            n_ap += 1
    n_ap = math.floor(n_ap / 2)
    # for index in range(0, len(x), 2):
    #     x, y = x[index], x[index + 1]
    #     if (x >= self.bounds['S_LB'] and x <= self.bounds['S_UB']) and (y >= self.bounds['S_LB']and y <= self.bounds['S_UB']):
    #         n_ap += 1

    return n_ap / self.dim


def _coperture(clients, x):
    covered = 0
    for client in clients:
        found = False
        index = 0
        while not found and index < len(x):
            dist = math.sqrt((client[0] - x[index]) ** 2 + (client[1] - x[index + 1]) ** 2)
            if dist <= radius:
                covered += 1
                found = True
            index += 2
    return -covered / len(self.clients)