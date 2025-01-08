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
    df = pd.read_csv(path, sep=r'\s+')
    return df

# data = sample_data()
path = "./data/baseball.dat" # 337 observation, 27 covariates + 1 target
data = get_data_dat(path)

GA = GA_variable_selector(data, target = "salary", gen_size=30, parent_method="prop_random", mutation_rate=0.02)
GA.select()
GA.show_optimal_code()
GA.plot_gen_vs_criterion()