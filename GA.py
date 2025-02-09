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
    def __init__(self, data, target, seed = None, gen_size = None, num_generations = 100, 
                 fitness_method:str = "rank", parent_method:str = "proportional", generation_gap = 1, 
                 crossover_method:str = "one_point", mutation_rate:float = None, 
                 tournament:bool = False, k:int = None,
                 method = "OLS", gls_matrix = None, criterion = "AIC", print_last_generation = False):
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
        self.G = generation_gap
        assert generation_gap <= 1 and generation_gap >= 1/self.gen_size, "Generation gap is too large or too small."

        # tournament selection
        self.tournament = tournament
        self.k = k

        # Safe the best model, criterion, and selected covariates
        self.best_model = None
        self.best_criterion = np.inf
        self.best_covariates = None

        # model and objective function (criterion)
        self.method = method
        self.GLS_matrix = gls_matrix
        self.criterion = criterion
        
        # Instance that saves the model in each generation
        self.model = [" "]*self.gen_size
        self.criterion_value =  [""]*self.gen_size
        self.fitness_score = [""]*self.gen_size # depends on criterion value but not necessarily equal

        # for plot analysis later
        self.all_criterion_value = []

        self.print_last_generation = print_last_generation

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
            offspring = []
            # within this generation, loop over each individual
            num_offspring = int(self.G*self.gen_size) # number of offspring to replace the population
            for _ in range(num_offspring):
                # find 2 parents based on the fitness score (default proportional to total rank fitness)
                parent = self.select_parent(tournament = self.tournament, k = self.k)
                # produce a offspring given two parents
                offspring.append(self.produce_offspring(parent))
            offspring = np.array(offspring)

            # Update the best selection on record
            if (min(self.criterion_value) < self.best_criterion):
                print("Best selection updated at generation ", i+1)
                index = self.criterion_value.index(min(self.criterion_value))
                self.best_criterion = min(self.criterion_value)
                self.best_model = self.model[index]
                best_cov = np.array(self.pop[index])
                self.best_covariates = np.where(best_cov == 1)[0]

            # Once all offspring is found, update population with the offspring 
            # For generation gap, deterministic version of elitist strategy is implemented
            if self.G == 1: # canonical genetic algorithm with nonoverlapping generations
                self.pop = offspring
            # if self.G == 1/self.gen_size, it means a steady state genetic algorithm with higher selective pressure and more variance
            else: # replace partial (least fit) population with offspring
                output_index = []
                topk_index = self.get_topN_index(self.fitness_score.copy(), output_index, num_offspring)
                self.pop[topk_index] = offspring
            
            # print the population after the last generation
            if (self.print_last_generation and i == self.num_generation - 1):
                print(i+1, "th generation")
                print(self.pop)
            i += 1

    def init_pop(self):
        '''
        initialize population with unique individuals
        '''
        assert self.gen_size <= 2**self.pop_size, "Exhausted all population, try a smaller value of generation size."

        self.pop = self.rng.choice(2, size = (self.gen_size, self.pop_size))
        pop = set(tuple(i) for i in self.pop)

        # generate unique individuals
        while len(pop) != self.gen_size:
            more_pop = self.rng.choice(2, size = (self.gen_size-len(pop), self.pop_size))
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
            if self.method == "WLS":
                ols_model = sm.OLS(self.y, ith_X).fit()
                residuals = ols_model.resid
                weight = 1/(residuals ** 2 + 1e-6)
                model = sm.WLS(self.y, ith_X, weight).fit()
                self.model[i] = model
            if self.method == "GLS":
                print("You would need to define the variance correlation matrix for GLS")
                model = sm.GLS(self.y, ith_X, self.GLS_matrix).fit()
                self.model[i] = model
            if self.method == "GLM":
                model = sm.GLM(self.y, ith_X, family = sm.families.Poisson()).fit()
                self.model[i] = model

    def eval_model(self):
        '''
        Evaluate all the models
        output: the criterion value for all individuals
        '''
        if self.criterion == "AIC": # akaike information criterion
            self.criterion_value  = [model.aic for model in self.model]
            self.all_criterion_value.append(self.criterion_value)
        if self.criterion == "BIC": # bayesian information criterion
            self.criterion_value  = [model.bic for model in self.model]
            self.all_criterion_value.append(self.criterion_value)
        if self.criterion == "LLF": # log likelihood function
            self.criterion_value  = [-model.llf for model in self.model]
            self.all_criterion_value.append(self.criterion_value)
        if self.criterion == "R2_adj":
            self.criterion_value  = [-model.rsquared_adj for model in self.model]
            self.all_criterion_value.append(self.criterion_value)
        if self.criterion == "MSE":
            self.criterion_value  = [model.mse_model for model in self.model]
            self.all_criterion_value.append(self.criterion_value)

    def calc_fitness_score(self):
        '''
        Offers proportional, ranking, and *tournament selection (increased selective power in this order)
        '''
        if self.fit_method == 'proportional':
            self.fitness_score = self.criterion_value/np.sum(self.criterion_value)
        elif self.fit_method == "rank":
            # highest criterion value has the lowest rank (at 1)
            rank = sp.stats.rankdata(-1*np.array(self.criterion_value), method = "ordinal")
            self.fitness_score = 2*rank/(self.gen_size*(self.gen_size + 1)) # median quality candidate a selection probability of 1/P, and the best chromosome has probability 2/(P + 1)
        
        assert abs(self.fitness_score.sum() - 1) < 1e-10, "fitness scores doesn't sum to 1 for rank based"

    def select_parent(self, tournament = False, k = None):
        # select two parents based on the fitness score
        if tournament == True:
            possible_parents = []
            # randomly partitioned into k disjoint subsets of equal size
            size = self.gen_size // k
            while len(possible_parents) < self.gen_size:
                # randomly split into k subsets
                shuffled_lst = self.rng.permutation(list(range(self.gen_size)))
                k_subsets = [shuffled_lst[i * size:(i + 1) * size].tolist() for i in range(k)]
                # select the best individual from each subset
                for subset in k_subsets:
                    max_index = np.argmax(self.fitness_score[subset])
                    possible_parents.append(subset[max_index])
            selection = self.rng.choice(possible_parents, size = 2)
            parent = self.pop[selection]
            return parent
        # proportional selection, ranking, and tournament selection apply increasing selective pressure
        if self.parent_method == "proportional":
            # select each parent independently with probability proportional to fitness
            selection = list(self.rng.choice(list(range(self.gen_size)), size = 2, p = self.fitness_score/self.fitness_score.sum()))
            parent = self.pop[selection]
            return parent
        elif self.parent_method == "prop_random":
            # select one parent with probability proportional to fitness and select the other parent completely at random
            s1 = self.rng.choice(list(range(self.gen_size)), p = self.fitness_score/self.fitness_score.sum())
            s2 = self.rng.choice(list(range(self.gen_size)))
            while s2 == s1:
                s2 = self.rng.choice(list(range(self.gen_size)))
            selection = [s1, s2]
            parent = self.pop[selection]
            return parent
            
    def produce_offspring(self, parent):
        assert len(parent) == 2, "There should be two parents"
        p1 = parent[0]
        p2 = parent[1]
        offspring = self.crossover(p1, p2)
        offspring = self.mutation(offspring)
        return offspring

    # Genetic Operators
    def crossover(self, p1, p2):
        assert len(p1) == len(p2), "Parents don't have the same length"
        if self.cross_method == 'one_point':
            n = len(p1) - 1
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
        elif self.cross_method == 'two_point':
            sep_points = np.sort(self.rng.choice(list(range(1, n)), size = 2))
            s1 = sep_points[0]
            s2 = sep_points[1]
            # Split parents
            p1_1 = p1[:s1]
            p1_2 = p1[s1:s2]
            p1_3 = p1[s2:]
            p2_1 = p2[:s1]
            p2_2 = p2[s1:s2]
            p2_3 = p2[s2:]
            # Form Offspring
            if self.rng.choice(2) == 1:
                off = np.append(p1_1, p2_2, p1_3)
            else:
                off = np.append(p2_1, p1_2, p2_3)
            return off
        
    def mutation(self, offspring:np.ndarray) -> np.ndarray:
        mutation_mask = self.rng.random(offspring.shape) < self.mutation_rate
        # when mask = True, use 1-offspring, else don't change and use the original offspring
        mutated_offspring = np.where(mutation_mask, 1 - offspring, offspring)
        return mutated_offspring

    # Helper functions
    def get_topN_index(self, input_lst:np.ndarray, output_lst:list, N:int) -> list:
        '''
        Get the top N least fit index in the population recursively
        input:
            - input_lst: input list (criterion values)
            - output_lst: output list (top N index)
            - N: number of top index to return
        '''
        if len(output_lst) == N:
            return output_lst
        min_index = np.argmin(input_lst)
        output_lst.append(min_index)
        input_lst[min_index] = np.inf
        return self.get_topN_index(input_lst, output_lst, N)

    # Optional plot for analysis
    def print_best_individual(self):
        criterion = np.array(self.all_criterion_value)
        criterion = criterion.flatten()
        print("\nBest criterion from genetic algorithm: ", criterion.min())
        print(self.best_covariates)

    def plot_gen_vs_criterion(self):
        criterion = np.array(self.all_criterion_value)
        gen = np.repeat(np.arange(self.num_generation), self.gen_size)  # number of generation, repeated gen_size times
        criterion = criterion.flatten()
        plt.scatter(gen, criterion, s=5, color="black")  # Small black dots
        plt.xlabel("Generation")
        plt.ylabel("Criterion Value")
        plt.title("Criterion vs. Generation")
        plt.show()

    # Optimal (according to book)
    def show_optimal(self):
        optimal_inds = [2,3,6,8,10,13,14,15,16,24,25,26]
        optimal_inds = [x - 1 for x in optimal_inds]
        filtered_X = self.X.iloc[:, optimal_inds]
        filtered_X = sm.add_constant(filtered_X)
        model = sm.GLM(self.y, filtered_X, family = sm.families.Poisson()).fit()
        print("\nTheoretically best criterion from the textbook: ", model.aic)
        print(optimal_inds)

        optimal_inds = [6, 7, 9, 10, 12, 13, 17, 24]
        filtered_X = self.X.iloc[:, optimal_inds]
        filtered_X = sm.add_constant(filtered_X)
        model = sm.GLM(self.y, filtered_X, family = sm.families.Poisson()).fit()
        print("\nTheoretically best criterion from past GA: ", model.aic)
        print(optimal_inds)