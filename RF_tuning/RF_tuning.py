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

# https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/

RESUME = False
NUM_COMPARISON = 15   # regret comparison in generated pdf

TIME_BUDGET_GP = 2800
TIME_BUDGET_RF = 4
MAX_ITERATIONS_GP = 1
MAX_ITERATIONS_RF = 1
DIM = 2
REPETITIONS = 2

NUMBER_TREE_VALUES = [0, 6]             # converted to 2^number_of_trees + 2 when testing
FEATURE_PERCENTAGE_VALUES = [0, 1]
BOOTSTRAP_VALUES = [0, 1]
MIN_SAMPLE_SPLIT_VALUES = [2, 10]       

# https://stats.stackexchange.com/questions/286107/setting-leaf-nodes-minimum-sample-value-for-random-forest-decision-trees ???
# https://bayesmark.readthedocs.io/en/latest/scoring.html#analyze-and-summarize-results

ACKLEY_RANGE = [-5, 5]
GRIEWANK_RANGE = [-5, 5]
RASTRIGIN_RANGE = [-5, 5]
SCHWEFEL_RANGE = [420.9687-5, 420.9687+5]

NUMBER_OF_TREES = [2, 5, 10, 25, 50, 100, 200]

run_no = 0

# configuration ranking for each function
# merge results to find best hyperparam
# min 4 score
# keep same scale on regret plots
# test on unkonown function

# black box optimization challenge 2022 arxiv
# SCORING https://bayesmark.readthedocs.io/en/latest/scoring.html

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


def function_max(function, dim, f_range):
    Xd = {}
    for i in range(dim):
        Xd["x" + str(i+1)] = f_range
    return function(Xd)

FUNCTIONS = [
            ["Ackley", ackley_function, [ACKLEY_RANGE] * DIM, ["value"], function_max(ackley_function, DIM, ACKLEY_RANGE[1])],
            ["Griewank", griewank_function, [GRIEWANK_RANGE] * DIM, ["value"], function_max(griewank_function, DIM, GRIEWANK_RANGE[1])],
            ["Rastrigin", rastrigin_function, [RASTRIGIN_RANGE] * DIM, ["value"], function_max(rastrigin_function, DIM, RASTRIGIN_RANGE[1])],
            ["Schwefel", schwefel_function, [SCHWEFEL_RANGE] * DIM, ["value"], function_max(schwefel_function, DIM, SCHWEFEL_RANGE[1])]
            ]
    

# random forest function: we ran random forest with the X parameters
def random_forest(X):
    number_of_trees = X["number_of_trees"]
    max_features = X["max_features"]
    bootstrap = X["bootstrap"]
    min_samples_split = X["min_samples_split"]   

    global run_no
    min = 0
    for f in FUNCTIONS:
        minf = 0
        for i in range(REPETITIONS):
            minf += random_forest_optimizer(f[0], f[1], f[2], f[4], number_of_trees, max_features, bootstrap, min_samples_split, run_no, i)
        minf /= REPETITIONS
        write_description_file(f[0], NUMBER_OF_TREES[int(number_of_trees)], float(max_features), bool(bootstrap), int(min_samples_split), minf, minf/f[4], run_no) 
        min += (minf/f[4])
    run_no += 1    
    return min/len(FUNCTIONS)



# random forest optimizer: run random forest bayesyan optimization and return the global minima
def random_forest_optimizer(function_name, function, values, max, number_of_trees, max_features, bootstrap, min_samples_split, run_no, rep_no):
    directory = "Tests/" + "dim" + str(DIM) + "/" + function_name + "/test" + str(run_no) + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    json_dir = directory + "scenario.json"
    csv_dir = directory + "output_samples_" + str(rep_no) + ".csv"

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
    scenario["models"]["number_of_trees"] = NUMBER_OF_TREES[int(number_of_trees)]
    scenario["models"]["max_features"] = float(max_features)
    scenario["models"]["bootstrap"] = bool(bootstrap)
    scenario["models"]["min_samples_split"] = int(min_samples_split)

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, function)

    # return the minimum value found by hypermapper
    df = pd.read_csv(csv_dir)
    min = float(df['value'].min())    
    return min
    

# run bayesyan optimization to tune random forest hyperparameters
def RF_hyperparameter_tuning(function):
    directory = "Optimizer/dim" + str(DIM) + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    csv_dir = directory + "optimizer_dim" + str(DIM) + ".csv"
    json_dir = directory + "optimizer_dim" + str(DIM) + ".json"

    scenario = {}    
    scenario["application_name"] = "RF optimizer"
    scenario["models"] = {}
    scenario["optimization_objectives"] = ["value"]
    scenario["optimization_iterations"] = MAX_ITERATIONS_GP
    scenario["time_budget"] = TIME_BUDGET_GP
    scenario["noise"] = True
    scenario["normalize_inputs"] = True
    scenario["output_data_file"] = csv_dir

    scenario["resume_optimization"] = False
    scenario["resume_optimization_data"] = "/mnt/d/Users/Luigi/Desktop/Universita/Tirocinio/RF_tuning/Optimizer/optimizer_dim" + str(DIM) + ".csv"
    
    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = 5   # d + 1

    scenario["input_parameters"] = {}
    
    number_of_trees = {}
    number_of_trees["parameter_type"] = "integer"             
    number_of_trees["values"] = NUMBER_TREE_VALUES   
    #number_of_trees["parameter_default"] = 3
    scenario["input_parameters"]["number_of_trees"] = number_of_trees

    max_features = {}
    max_features["parameter_type"] = "real"               
    max_features["values"] = FEATURE_PERCENTAGE_VALUES     
    #max_features["parameter_default"] = 0.5
    scenario["input_parameters"]["max_features"] = max_features      

    bootstrap = {}
    bootstrap["parameter_type"] = "integer"            
    bootstrap["values"] = BOOTSTRAP_VALUES
    #bootstrap["parameter_default"] = 1
    scenario["input_parameters"]["bootstrap"] = bootstrap

    min_samples_split = {}
    min_samples_split["parameter_type"] = "integer"            
    min_samples_split["values"] = MIN_SAMPLE_SPLIT_VALUES
    #min_samples_split["parameter_default"] = 5
    scenario["input_parameters"]["min_samples_split"] = min_samples_split      

    scenario["models"]["model"] = "gaussian_process"
    scenario["acquisition_function"] = "EI"

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, function)
    sys.stdout = stdout


# write a file with test info in test directory
def write_description_file(f_name, number_of_trees, max_features, bootstrap, min_samples_split, min, norm_min, run_no):
    dir = "Tests/" + "dim" + str(DIM) + "/" + f_name + "/tests_specs.csv"
    with open(dir, "a+", newline='') as write_obj:
        if run_no == 0:
            csv.writer(write_obj).writerow(["test_number", "number_of_trees", "max_features", "bootstrap", "min_samples_split", "mean_min", "mean_normalized_min"])
        csv.writer(write_obj).writerow([run_no, number_of_trees, max_features, bootstrap, min_samples_split, min, norm_min])


# plot optimization results (regret)
def plot_optimization(f_name):
    dirs = "Tests/" + "dim" + str(DIM) + "/" + f_name + "/"
    json_dir = dirs + "test0/" + f_name + "_scenario.json"
    output_file = dirs + "regret.pdf"

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



RF_hyperparameter_tuning(random_forest)