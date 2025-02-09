from GA import *

def sample_data(size = 100):
    rng  = np.random.default_rng()
    x1 = rng.choice(1000, size) # relevant
    x2 = rng.choice(1000, size)
    x3 = rng.choice(1000, size) 
    x4 = rng.choice(1000, size)
    x5 = rng.choice(1000, size)
    x6 = rng.choice(1000, size)
    noise = rng.standard_normal(size)
    y = 4*x1 + 2 + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})

def get_data_dat(path:str) -> pd.DataFrame:
    '''
    Takes in a path to a dat file, read it and save it as dataframe
    '''
    if path[-4:] == ".dat":
        df = pd.read_csv(path, sep=r'\s+')
    elif path[-4:] == ".csv":
        df = pd.read_csv(path)
    return df

# data = sample_data()
path = "./data/baseball.dat" # 337 observation, 27 covariates + 1 target
data = get_data_dat(path)

GA = GA_variable_selector(data, target = "salary", seed = None, gen_size=30, num_generations = 80, 
                          fitness_method = "rank", parent_method="prop_random", generation_gap = 20/30, 
                          crossover_method = "one_point", mutation_rate = 0.02, 
                          tournament = True, k = 3,
                          method = "GLM", criterion = "MSE", print_last_generation = False)
GA.select()
GA.show_optimal()
GA.print_best_individual()
GA.plot_gen_vs_criterion()