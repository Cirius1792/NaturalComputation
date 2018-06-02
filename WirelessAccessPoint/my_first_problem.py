import pygmo as pg

class my_problem:
    def __init__(self, dim=10):
        self.dim = dim


    def fitness(self, x):
        """The fitness() method is expected to return the fitness of the input decision vector """
        return [sum(x * x)]



    def get_bounds(self):
        """ get_bounds() is expected to return the box bounds of the problem, (lb,ub), which also implicitly define the dimension of the problem"""
        return ([-1] * self.dim, [1] * self.dim)


if __name__ == '__main__':
    prob = pg.problem(my_problem(3))
    print(prob)
    #class pygmo.de(gen = 1, F = 0.8, CR = 0.9, variant = 2, ftol = 1e-6, xtol = 1e-6, seed = random)
    gen = 50
    F = 0.8
    CR = 0.9
    seed = 7
    algo = pg.algorithm(pg.de(gen,F,CR))
    pop = pg.population(prob, 10)
    pop = algo.evolve(pop)
    print(pop.champion_f)
