import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hypermapper import optimizer
stdout = sys.stdout 



HP = {
        "acquisition_function": "UCB",
        "local_search_starting_points": 0,
        "local_search_random_points": 0,
        "log_transform_output": 0,
        "lenghtscale_prior": 0,
        "epsilon_greedy_threshold": 0,
        "predict_noiseless": 0,
        "exploration_augmentation": -5
    }


ACKLEY_RANGE = [-5, 5]
GRIEWANK_RANGE = [-5, 5]
RASTRIGIN_RANGE = [-5, 5]
SCHWEFEL_RANGE = [420.9687-5, 420.9687+5]

ITERATIONS = 38

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


# RUNNING TESTS
def run_test(function_name, function, global_minimum, values, optimization_objectives, run_no):
    dim = len(values)
    GPu_dir = "GPvsGP/" + "dim" + str(dim) + "/" + function_name + "/Untuned/"
    GPt_dir = "GPvsGP/" + "dim" + str(dim) + "/" + function_name + "/Tuned/"
    if not os.path.exists(GPu_dir): os.makedirs(GPu_dir)
    if not os.path.exists(GPt_dir): os.makedirs(GPt_dir)

    GPu_json = GPu_dir + "GPu_" + function_name + "_scenario.json"
    GPt_json = GPt_dir + "GPt_" + function_name + "_scenario.json"  
    GPu_csv = GPu_dir + "GPu_output_samples_" + str(run_no) + ".csv"
    GPt_csv = GPt_dir + "GPt_output_samples_" + str(run_no) + ".csv"  
    
    scenario = {}    
    scenario["models"] = {}
    scenario["optimization_objectives"] = optimization_objectives
    scenario["optimization_iterations"] = ITERATIONS
    scenario["noise"] = True
    scenario["normalize_inputs"] = True
    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = dim + 1

    scenario["input_parameters"] = {}
    for i in range(dim):
        x = {}
        x["parameter_type"] = "real"
        x["values"] = values[i]
        scenario["input_parameters"]["x" + str(i+1)] = x    

    scenario["application_name"] = "GP_" + function_name
    scenario["models"]["model"] = "gaussian_process"
    scenario["output_data_file"] = GPu_csv    

    with open(GPu_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)

    scenario["application_name"] = "GP_" + function_name + "_tuned"
    scenario["models"]["model"] = "gaussian_process"      
    scenario["models"]["acquisition_function"] = HP["acquisition_function"]
    scenario["models"]["local_search_starting_points"] = HP["local_search_starting_points"]
    scenario["models"]["local_search_random_points"] = HP["local_search_random_points"]
    scenario["models"]["log_transform_output"] = HP["log_transform_output"]
    scenario["models"]["lengthscale_prior"] = {}
    scenario["models"]["lengthscale_prior"]["name"] = "gamma"
    scenario["models"]["lengthscale_prior"]["parameters"] = [1.3, 0.1]
    scenario["models"]["epsilon_greedy_threshold"] = HP["epsilon_greedy_threshold"]
    scenario["models"]["predict_noiseless"] = HP["predict_noiseless"]
    scenario["models"]["exploration_augmentation"] = HP["exploration_augmentation"]
    scenario["output_data_file"] = GPt_csv

    with open(GPt_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(GPu_json, function)
    optimizer.optimize(GPt_json, function)
    sys.stdout = stdout


def get_function_lists(dim):    
    functions = list()
    functions.append(["Ackley", ackley_function, [0] * (dim+1), [ACKLEY_RANGE] * dim, ["value"], 0])
    functions.append(["Griewank", griewank_function, [0] * (dim+1), [GRIEWANK_RANGE] * dim, ["value"], 0])
    functions.append(["Rastrigin", rastrigin_function, [0] * (dim+1), [RASTRIGIN_RANGE] * dim, ["value"], 0])
    functions.append(["Schwefel", schwefel_function, [*list([420.9687]*dim), 0], [SCHWEFEL_RANGE] * dim, ["value"], 0])
    return functions


# PLOT OPTIMIZATION RESULTS
def plot_optimization(function_name, dim):
    # create file sh for plotting optimization results
    f = open("plot_optimization.sh", "w+")
    f.write('hm-plot-optimization-results -j $1 -i $2 $3 -o $4 -t "$5" -l "$6" "$7" -log')
    f.close()
    # create directory if not exists
    output_dir ="GPvsGP/dim" + str(dim) + "/" + function_name +  "/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # run sh
    file_dir = output_dir + "Untuned/GPu_" + function_name + "_scenario.json"
    output_file = output_dir + "optimization_results" + str(dim) + ".pdf"
    subprocess.Popen(["bash", "plot_optimization.sh", file_dir, output_dir + "Untuned/", output_dir + "Tuned/", output_file, r'\rm{Regret ' + function_name + "}", r'\rm{GP default}', r'\rm{GP tuned}'])


# MAIN
def main(dim, n_executions):
    # get function list
    functions = get_function_lists(dim)

    # for each function, run tests and plot optimization results
    for f in functions:
        for _ in range(n_executions):
            #run_test(f[0], f[1], f[2], f[3], f[4], f[5])
            f[5] += 1 
        plot_optimization(f[0], dim)


# dimension, executions
main(2, 4)