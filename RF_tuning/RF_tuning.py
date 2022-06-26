import os
import sys
import csv
import json
import subprocess
import numpy as np
import pandas as pd
from math import ceil
from hypermapper import optimizer
stdout = sys.stdout 

RESUME = False

NUM_COMPARISON = 5

TIME_BUDGET_GP = 1500
TIME_BUDGET_RF = 30
MAX_ITERATIONS_GP = 50
MAX_ITERATIONS_RF = 200
DIM = 12

TREE_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
FEATURE_PERCENTAGE_VALUES = [0, 1]
BOOTSTRAP_VALUES = [0, 1]
SAMPLE_SPLIT_VALUES = [2, 10]

ACKLEY_RANGE = [-5, 5]
GRIEWANK_RANGE = [-5, 5]
RASTRIGIN_RANGE = [-5, 5]
SCHWEFEL_RANGE = [-5, 5]

ackley_run_n = 0
griewank_run_n = 0
rastrigin_run_n = 0
schwefel_run_n = 0


# Nd FUNCTIONS

def ackley_function(Xd):
    X = []
    for x in Xd:
        X.append(Xd[x])
    X = np.array(X)
    a, b, c = 20, 0.2, 2*np.pi
    return - a * np.exp(-b * np.sqrt(np.sum(np.mean(X**2)))) - np.exp(np.sum(np.mean(np.cos(c * X)))) + a + np.exp(1)

def griewank_function(Xd):
    X = []
    for x in Xd:
        X.append(Xd[x])
    X = np.array(X)
    sqrti = np.sqrt(np.array(list(range(1, len(X)+1))))
    return 1 + np.sum(X**2 / 4000) - np.prod(np.cos(X / sqrti))

def rastrigin_function(Xd):
    X = []
    for x in Xd:
        X.append(Xd[x])
    X = np.array(X)
    return 10 * len(X) + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))

def schwefel_function(Xd):
    X = []
    for x in Xd:
        X.append(Xd[x])
    X = np.array(X)
    return 418.9829 * len(X) - np.sum(X * np.sin(np.sqrt(np.abs(X))))



# random forest function: we ran random forest with the X parameters

def random_forest_ackley(X):
    number_of_trees = X["number_of_trees"]
    max_features = X["max_features"]
    bootstrap = X["bootstrap"]
    min_samples_split = X["min_samples_split"]   
    
    global ackley_run_n
    min_found = random_forest_optimizer("Ackley", ackley_function, [ACKLEY_RANGE] * DIM, number_of_trees, max_features, bootstrap, min_samples_split, ackley_run_n)
    ackley_run_n = ackley_run_n + 1
    return min_found

def random_forest_griewank(X):
    number_of_trees = X["number_of_trees"]
    max_features = X["max_features"]
    bootstrap = X["bootstrap"]
    min_samples_split = X["min_samples_split"]   
    
    global griewank_run_n
    min_found = random_forest_optimizer("Griewank", griewank_function, [GRIEWANK_RANGE] * DIM, number_of_trees, max_features, bootstrap, min_samples_split, griewank_run_n)
    griewank_run_n = griewank_run_n + 1
    return min_found

def random_forest_rastrigin(X):
    number_of_trees = X["number_of_trees"]
    max_features = X["max_features"]
    bootstrap = X["bootstrap"]
    min_samples_split = X["min_samples_split"]   
      
    global rastrigin_run_n
    min_found = random_forest_optimizer("Rastrigin", rastrigin_function, [RASTRIGIN_RANGE] * DIM, number_of_trees, max_features, bootstrap, min_samples_split, rastrigin_run_n)
    rastrigin_run_n = rastrigin_run_n + 1
    return min_found

def random_forest_schwefel(X):
    number_of_trees = X["number_of_trees"]
    max_features = X["max_features"]
    bootstrap = X["bootstrap"]
    min_samples_split = X["min_samples_split"]   

    global schwefel_run_n
    min_found = random_forest_optimizer("Schwefel", schwefel_function, [SCHWEFEL_RANGE] * DIM, number_of_trees, max_features, bootstrap, min_samples_split, schwefel_run_n)
    schwefel_run_n = schwefel_run_n + 1
    return min_found


# random forest optimizer: run random forest bayesyan optimization and return the global minima
def random_forest_optimizer(function_name, function, values, number_of_trees, max_features, bootstrap, min_samples_split, run_no):
    directory = "Tests/" + function_name + "/test" + str(run_no) + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    json_dir = directory + function_name + "_scenario.json"
    csv_dir = directory + function_name + "_output_samples.csv"

    scenario = {}    
    scenario["application_name"] = "RF_" + function_name
    scenario["models"] = {}
    scenario["optimization_objectives"] = ["value"]
    scenario["optimization_iterations"] = MAX_ITERATIONS_RF
    scenario["time_budget"] = TIME_BUDGET_RF
    scenario["noise"] = True
    scenario["normalize_inputs"] = True
    scenario["output_data_file"] = csv_dir
    scenario["resume_optimization"] = RESUME
    scenario["resume_optimization_data"] = csv_dir

    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = len(values) + 1    # d+1 for initial random sampling
    scenario["input_parameters"] = {}
    for i in range(len(values)):
        x = {}
        x["parameter_type"] = "real"
        x["values"] = values[i]
        scenario["input_parameters"]["x" + str(i+1)] = x        

    scenario["models"]["model"] = "random_forest"
    scenario["models"]["number_of_trees"] = int(number_of_trees)
    scenario["models"]["max_features"] = float(max_features)
    scenario["models"]["bootstrap"] = bool(bootstrap)
    scenario["models"]["min_samples_split"] = int(min_samples_split)

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, function)
    # sys.stdout = stdout

    # return the minimum value found by hypermapper
    df = pd.read_csv(csv_dir)
    min = df['value'].min()
    write_description_file(function_name, int(number_of_trees), float(max_features), bool(bootstrap), int(min_samples_split), float(min), run_no) 
    return min
    

# run bayesyan optimization to tune random forest hyperparameters
def RF_hyperparameter_tuning(random_forest_f, f_name):
    directory = "Optimizer/"
    if not os.path.exists(directory): os.makedirs(directory)

    csv_dir = directory + "optimizer.csv"
    json_dir = directory + "optimizer.json"

    scenario = {}    
    scenario["application_name"] = "RF optimizer"
    scenario["models"] = {}
    scenario["optimization_objectives"] = ["value"]
    scenario["optimization_iterations"] = MAX_ITERATIONS_GP
    scenario["time_budget"] = TIME_BUDGET_GP
    scenario["noise"] = True
    scenario["normalize_inputs"] = True
    scenario["output_data_file"] = csv_dir
    
    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = 5

    scenario["input_parameters"] = {}
    
    number_of_trees = {}
    number_of_trees["parameter_type"] = "ordinal"             
    number_of_trees["values"] = TREE_VALUES    
    scenario["input_parameters"]["number_of_trees"] = number_of_trees

    max_features = {}
    max_features["parameter_type"] = "real"               
    max_features["values"] = FEATURE_PERCENTAGE_VALUES
    scenario["input_parameters"]["max_features"] = max_features      

    bootstrap = {}
    bootstrap["parameter_type"] = "integer"            
    bootstrap["values"] = BOOTSTRAP_VALUES
    scenario["input_parameters"]["bootstrap"] = bootstrap

    min_samples_split = {}
    min_samples_split["parameter_type"] = "integer"            
    min_samples_split["values"] = SAMPLE_SPLIT_VALUES
    scenario["input_parameters"]["min_samples_split"] = min_samples_split      

    scenario["models"]["model"] = "gaussian_process"
    scenario["acquisition_function"] = "EI"
    scenario["output_data_file"] = csv_dir 

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, random_forest_f)

    f_dir = "Tests/" + f_name + "/"
    plot_optimization(f_dir + "test0/" + f_name + "_scenario.json", f_dir, f_dir + "optimization_results.pdf", f_name)
    

    sys.stdout = stdout


# write a file with test info in test directory
def write_description_file(f_name, number_of_trees, max_features, bootstrap, min_samples_split, min, run_no):
    dir = "Tests/" + f_name + "/tests_specs.csv"
    with open(dir, "a+", newline='') as write_obj:
        if run_no == 0:
            csv.writer(write_obj).writerow(["test_number", "number_of_trees", "max_features", "bootstrap", "min_samples_split", "min",])
        csv.writer(write_obj).writerow([run_no, number_of_trees, max_features, bootstrap, min_samples_split, min])


def write_min_in_csv(dir):
    df = pd.read_csv(dir + "tests_specs.csv")
    minimum = df.iloc[[df['min'].idxmin()]]
    f = open(dir + "minima.txt", "w")
    f.write(str(minimum))
    f.close()


# plot optimization results (regret)
def plot_optimization(json_dir, dirs, output_file, f_name):
    code = 'hm-plot-optimization-results -j ' + json_dir + ' -i '
    folders = [name for name in os.listdir(dirs) if os.path.isdir(os.path.join(dirs, name))]
    folders = folders[0::(ceil(len(folders)/NUM_COMPARISON))]
    for folder in folders:
        code += dirs + str(folder) + " " 
    code += '-o ' + output_file + ' -t "$1" -l '
    for folder in folders:
        code += '"\\rm{' + folder + '}" '

    # create file sh for plotting optimization results
    f = open("plot_optimization.sh", "w+")
    f.write(code)
    f.close()
    subprocess.Popen(["bash", "plot_optimization.sh", r'\rm{' + f_name + ' Regret}'])



f_list = [(random_forest_griewank, "Griewank"), (random_forest_rastrigin, "Rastrigin"), (random_forest_schwefel, "Schwefel")] 

for f in f_list:
    RF_hyperparameter_tuning(f[0], f[1])
    write_min_in_csv("Tests/" + f[1] + "/")