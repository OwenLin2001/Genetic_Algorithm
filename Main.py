import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    def __init__(self, data, target, gen_size = None, num_generations = 20, method = "OLS", criterion = "AIC"):
        self.X = data.drop(target, axis = 1)
        self.y = data[target]
        self.pop_size = self.X.shape[1]  # C-chromosome length
        if gen_size == None:
            self.gen_size = 2*self.X.shape[1]  # P-size of generation, 2C as the default
        else:
            self.gen_size = gen_size

        self.method = method
        self.criterion = criterion

        self.model = [" "]*8
        # self.criterion_value =  [""]*gen_size

        self.num_generation = num_generations

    def select(self):
        self.init_pop()
        self.run_model()
        self.eval_model()
        i = 1
        while i < self.num_generation:
            i += 1

    def init_pop(self):
        np.random.seed(174)
        self.pop = np.random.randint(2, size = (self.gen_size, self.pop_size))

    def run_model(self):
        # Runing the model with selected covariates
        for i, individual in enumerate(self.pop):
            covariate = (individual == 1)
            ith_X = self.X.iloc[:,covariate]
            ith_X = sm.add_constant(ith_X)
            if self.method == "OLS":
                model = sm.OLS(self.y, ith_X).fit()
                self.model[i] = model
    
    def eval_model(self):
        # Use given criterion
        if self.criterion == "AIC":
            criterion_value  = [model.aic for model in self.model]
        print(criterion_value)

    def rank_fitness(self):
        pass

    def calc_fitness_score(self):
        pass

    def select_parent(self):
        parent = None
        self.pop = parent

# import os
# cwd = os.getcwd()
# data = pd.read_csv(cwd+r"\data\baseball.dat", sep=" ",header = 0)
# 337 observation, 27 covariates + 1 target


def sample_data():
    x1 = np.random.normal(0, 1, 50)
    x2 = np.random.normal(3, 1, 50)
    x3 = np.random.normal(-1, 1, 50)
    x4 = np.random.normal(0, 4, 50)
    eps = np.random.normal(0, 0.5, 50)
    y = x1 + x2 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4})

data = sample_data()

GA = GA_variable_selector(data, target = "y")
GA.select()
print(GA.pop)