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
    def __init__(self, data, target, seed = None, gen_size = None, num_generations = 100, mutation_rate:float = None, 
                 fitness_method:str = "rank", parent_method:str = "proportional", crossover_method:str = "simple",
                 method = "OLS", criterion = "AIC"):
        # Data separation
        self.X = data.drop(target, axis = 1)
        self.y = data[target]

        # random number generator
        self.rng = np.random.default_rng(seed)
        
        # Define Generation, a length p list, each element is a list of length C
        self.pop_size = self.X.shape[1]  # C-chromosome length
        if gen_size == None:
            self.gen_size = 2*self.X.shape[1]  # P-size of generation, 2C as the default
        else:
            self.gen_size = gen_size
        
        # Define how to run a genetic method
        if mutation_rate == None:
            self.mutation_rate = 1/self.pop_size
            self.mutation_rate = 1/self.gen_size/np.sqrt(self.pop_size)
        else:
            self.mutation_rate = mutation_rate
        self.fit_method = fitness_method
        self.parent_method = parent_method
        self.cross_method = crossover_method
        self.num_generation = num_generations

        # Safe the best model, criterion, and selected covariates
        self.best_model = None
        self.best_criterion = np.inf
        self.best_covariates = None

        # model and objective function (criterion)
        self.method = method
        self.criterion = criterion
        
        # Instance that saves the model in each generation
        self.model = [" "]*self.gen_size
        self.criterion_value =  [""]*self.gen_size
        self.fitness_score = [""]*self.gen_size # depends on criterion value but not necessarily equal

        # for plot analysis later
        self.all_criterion_value = []

    def select(self):
        # initialize a population with a given generation size
        self.init_pop()
        print("The population has size ", self.pop.shape)
        print()
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
                offspring[j] = self.produce_offspring(parent)

            # Update the best selection on record
            if (min(self.criterion_value) < self.best_criterion):
                print("Best selection updated.")
                index = self.criterion_value.index(min(self.criterion_value))
                self.best_criterion = min(self.criterion_value)
                self.best_model = self.model[index]
                best_cov = np.array(self.pop[index])
                self.best_covariates = np.where(best_cov == 1)[0]

            # Once all offspring is found, update population with the offspring
            self.pop = offspring
            if (i == 99):
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
            self.fitness_score = self.criterion_value/np.sum(self.criterion_value)
        elif approach == "rank":
            # highest criterion value has the lowest rank (at 1)
            rank = sp.stats.rankdata(-1*np.array(self.criterion_value), method = "ordinal")
            self.fitness_score = 2*rank/(self.gen_size*(self.gen_size + 1)) # median quality candidate a selection probability of 1/P, and the best chromosome has probability 2/(P + 1)
        
        assert abs(self.fitness_score.sum() - 1) < 1e-10, "fitness scores doesn't sum to 1 for rank based"

    def select_parent(self, method = "proportional", k = None):
        # select two parents based on the fitness score
        # proportional selection, ranking, and tournament selection apply increasing selective pressure
        if method == "proportional":
            # select each parent independently with probability proportional to fitness
            selection = list(self.rng.choice(list(range(self.gen_size)), size = 2, p = self.fitness_score/self.fitness_score.sum()))
            parent = self.pop[selection]
            return parent
        elif method == "prop_random":
            # select one parent with probability proportional to fitness and select the other parent completely at random
            s1 = self.rng.choice(list(range(self.gen_size)), p = self.fitness_score/self.fitness_score.sum())
            s2 = self.rng.choice(list(range(self.gen_size)))
            while s2 == s1:
                s2 = self.rng.choice(list(range(self.gen_size)))
            selection = [s1, s2]
            parent = self.pop[selection]
            return parent
        elif method == "tournament":
            # randomly partitioned into k disjoint subsets of equal size
            size = self.gen_size // k
            

            return parent
            
    def produce_offspring(self, parent):
        p1 = parent[0]
        p2 = parent[1]
        offspring = self.crossover(p1, p2)
        offspring = self.mutation(offspring, self.mutation_rate)
        return offspring

    # Genetic Operators
    def crossover(self, p1, p2, method = 'simple'):
        assert len(p1) == len(p2), "Parents don't have the same length"
        if method == 'simple':
            n = len(p1)
            sep_point = self.rng.choice(list(range(1, n)))
            # Split parents
            p1_1 = p1[:sep_point]
            p1_2 = p1[sep_point:]
            p2_1 = p2[:sep_point]
            p2_2 = p2[sep_point:]
            # Form Offspring - can also add the other pair for speed
            if self.rng.choice(2) == 1:
                off = np.append(p1_1, p2_2)
            else:
                off = np.append(p2_1, p1_2)
            assert off.shape == p1.shape, "offspring has a different shape than its parents"
            return off
        
    def mutation(self, offspring:np.ndarray, mutation_rate:float) -> np.ndarray:
        mutation_mask = self.rng.random(offspring.shape) < mutation_rate
        # when mask = True, use 1-offspring, else don't change and use the original offspring
        mutated_offspring = np.where(mutation_mask, 1 - offspring, offspring)
        return mutated_offspring

    # Optional plot for analysis
    def plot_gen_vs_criterion(self):
        criterion = np.array(self.all_criterion_value)
        gen = np.repeat(np.arange(self.num_generation), self.gen_size)  # number of generation, repeated gen_size times
        criterion = criterion.flatten()
        print("\nBest AIC from genetic algorithm: ", criterion.min())
        print(self.best_covariates)
        plt.scatter(gen, criterion, s=5, color="black")  # Small black dots
        plt.xlabel("Generation")
        plt.ylabel("Criterion Value")
        plt.title("Criterion vs. Generation")
        plt.show()

    # Optimal (according to book)
    def show_optimal_code(self):
        optimal_inds = [2,3,6,8,10,13,14,15,16,24,25,26]
        optimal_inds = [x - 1 for x in optimal_inds]
        filtered_X = self.X.iloc[:, optimal_inds]
        model = sm.OLS(self.y, filtered_X).fit()
        print("\nTheoretically best AIC from the textbook: ", model.aic)
        print(optimal_inds)