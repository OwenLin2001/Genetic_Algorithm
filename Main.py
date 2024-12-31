import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt
'''
Structure:
1. Initialize population
2. Check for the fitness score of each population
    - define fitneses score by criterion
    - order based on ranking (see book)
3. Choose the better fit few as parents !!!
4. Generate new population 

'''

class GA_variable_selector:
    def __init__(self, data, target, seed = None, gen_size = None, num_generations = 100, method = "OLS", criterion = "AIC"):
        # Data separation
        self.X = data.drop(target, axis = 1)
        self.y = data[target]

        # random number generator
        self.rng = np.random.default_rng(seed)
        
        # Definition Generation
        self.pop_size = self.X.shape[1]  # C-chromosome length
        if gen_size == None:
            self.gen_size = 2*self.X.shape[1]  # P-size of generation, 2C as the default
        else:
            self.gen_size = gen_size
        
        # model and objective function (criterion)
        self.method = method
        self.criterion = criterion
        
        self.model = [" "]*self.gen_size
        self.criterion_value =  [""]*self.gen_size
        self.fitness_score = [""]*self.gen_size # depends on criterion value but not necessarily equal

        self.num_generation = num_generations

        # for plot analysis later
        self.all_criterion_value = []

    def select(self):
        # initialize a population with a given generation size
        self.init_pop()
        print("The population has size ", self.pop.shape)
        i = 0
        # for each generation
        while i < self.num_generation:
            # create a model for each individual
            self.construct_model()
            # return the selected criterion as a measure of fit (default is AIC)
            self.eval_model()
            # convert the criterion to actual fitness score (default is rank)
            self.calc_fitness_score()
            # create a temporary offspring that replace pop later on
            offspring = np.zeros_like(self.pop)
            # within this generation, loop over each individual
            for j in range(self.gen_size):
                # find 2 parents based on the fitness score (default proportional to total rank fitness)
                parent = self.select_parent()
                # produce a offspring given two parents
                offspring[j] = self.produce_offspring(parent, self.rng)
            # Once all offspring is found, update population with the offspring
            self.pop = offspring
            if (i%20 == 0):
                print(i+1, "th generation")
                print(self.pop)
            i += 1

    def init_pop(self):
        '''
        initialize population with unique individuals
        '''
        assert self.gen_size <= 2**self.pop_size, "Exhausted all population, try a smaller value of generation size."

        self.pop = np.random.choice(2, size = (self.gen_size, self.pop_size))
        pop = set(tuple(i) for i in self.pop)

        # generate unique individuals
        while len(pop) != self.gen_size:
            more_pop = np.random.choice(2, size = (self.gen_size-len(pop), self.pop_size))
            # convert to set
            more_pop = set(tuple(i) for i in more_pop)
            pop = pop.union(more_pop)

            if len(pop) >= self.gen_size:
                pop = list(pop)[:self.gen_size]
        # Save in attribute
        for i,indiv in enumerate(pop):
            self.pop[i] = list(indiv)
        print("Starting Generation:")
        print(self.pop)

    def construct_model(self):
        '''
        Run the model for all P individuals with corresponding selected covariate
        output: the model for all individuals
        '''
        for i, individual in enumerate(self.pop):
            covariate = (individual == 1)
            ith_X = self.X.iloc[:, covariate]
            ith_X = sm.add_constant(ith_X)
            if self.method == "OLS":
                model = sm.OLS(self.y, ith_X).fit()
                self.model[i] = model
    
    def eval_model(self):
        '''
        Evaluate all the models
        output: the criterion value for all individuals
        '''
        if self.criterion == "AIC":
            self.criterion_value  = [model.aic for model in self.model]
            self.all_criterion_value.append(self.criterion_value)

    def calc_fitness_score(self, approach = "rank"):
        '''
        Offers proportional, ranking, and *tournament selection (increased selective power in this order)
        '''
        if approach == 'proportional':
            pass
        elif approach == "rank":
            # highest criterion value has the lowest rank (at 1)
            rank = sp.stats.rankdata(-1*np.array(self.criterion_value), method = "ordinal")
            self.fitness_score = 2*rank/(self.gen_size*(self.gen_size + 1)) # median quality candidate a selection probability of 1/P, and the best chromosome has probability 2/(P + 1)
            # print(self.criterion_value)
            # print(rank)
            # print(self.fitness_score)
            assert abs(self.fitness_score.sum() - 1) < 1e-10, "fitness scores doesn't sum to 1 for rank based"

    def select_parent(self, method = "proportional"):
        if method == 'tournament':
            pass
        elif method == "proportional":
            # select each parent independently with probability proportional to fitness
            selection = list(self.rng.choice(list(range(self.gen_size)), size = 2, p = self.fitness_score/self.fitness_score.sum()))
            parent = self.pop[selection]
            return parent
        elif method == "prop_random":
            # select one parent with probability proportional to fitness and select the other parent completely at random
            pass
            
    def produce_offspring(self, parent, rng):
        p1 = parent[0]
        p2 = parent[1]
        offspring = self.crossover(p1, p2, rng)
        offspring = self.mutation(offspring)
        return offspring

    # Genetic Operators
    def crossover(self, p1, p2, rng, method = 'simple'):
        assert len(p1) == len(p2), "Parents don't have the same length"
        if method == 'simple':
            n = len(p1)
            sep_point = rng.choice(list(range(1, n)))
            # Split parents
            p1_1 = p1[:sep_point]
            p1_2 = p1[sep_point:]
            p2_1 = p2[:sep_point]
            p2_2 = p2[sep_point:]
            # Form Offspring - can also add the other pair for speed
            if rng.choice(2) == 1:
                off = np.append(p1_1, p2_2)
            else:
                off = np.append(p2_1, p1_2)
            assert off.shape == p1.shape, "offspring has a different shape than its parents"
            return off
        
    def mutation(self, offspring:np.ndarray, mutation_rate:float = 0.01) -> np.ndarray:
        mutation_mask = self.rng.random(offspring.shape) < mutation_rate
        # when mask = True, use 1-offspring, else don't change and use the original offspring
        mutated_offspring = np.where(mutation_mask, 1 - offspring, offspring)
        return mutated_offspring

    # Optional plot for analysis
    def plot_gen_vs_criterion(self):
        criterion = np.array(self.all_criterion_value)
        gen = np.repeat(np.arange(self.num_generation), self.gen_size)  # number of generation, repeated gen_size times
        criterion = criterion.flatten()
        plt.scatter(gen, criterion, s=5, color="black")  # Small black dots
        plt.xlabel("Generation")
        plt.ylabel("Criterion Value")
        plt.title("Criterion vs. Generation")
        plt.show()

# import os
# cwd = os.getcwd()
# data = pd.read_csv(cwd+r"\data\baseball.dat", sep=" ",header = 0)
# 337 observation, 27 covariates + 1 target


def sample_data(size = 100):
    rng  = np.random.default_rng()
    x1 = rng.choice(1000, size) # relevant
    x2 = rng.choice(1000, size)
    x3 = rng.choice(1000, size) 
    x4 = rng.choice(1000, size)
    x5 = rng.choice(1000, size)
    x6 = rng.choice(1000, size)
    noise = rng.standard_normal(size)
    y = 4 * x1 + 2 + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})

data = sample_data()

GA = GA_variable_selector(data, target = "y")
GA.select()
GA.plot_gen_vs_criterion()