from matplotlib.pyplot import plot, show
from matplotlib import pyplot as plt


class APPlotter:
    def __init__(self, clients):
        self.clients = clients
        for ant in clients:
            plot(ant[0], ant[1], 'or')
        # #show(block=False)
        # plt.pause(0.5)
        # self.plots = []

    # def update(self, memset):
    #     for p in self.plots:
    #         if p is not None:
    #             p[0].pop(0).remove()
    #             p[1].remove()
    #     self.plots = [None]*len(memset)
    #
    #     for k in range(len(memset)):
    #         i = memset[k]
    #         circle1 = plt.Circle(i.paratopes, i.radius, color='green', alpha=0.3)
    #     plt.pause(0.5)

    def update(self, aps, radius):
        for index in range(0,len(aps), 2):
            x,y = aps[index], aps[index+1]
            circle = plt.Circle((x,y), radius, color='green', alpha=0.3)
            plt.gcf().gca().add_artist(circle)
        show(block=False)
        plt.pause(0.5)

    def _prepare_solution(self, aps):
        pass