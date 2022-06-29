import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hypermapper import optimizer
stdout = sys.stdout 

ACKLEY_HPO = {"number_of_trees": 4, "max_features": 0.705391, "bootstrap": True, "min_samples_split": 2}
GRIEWANK_HPO = {"number_of_trees": 256, "max_features": 0.454725, "bootstrap": True, "min_samples_split": 3}
RASTRIGIN_HPO = {"number_of_trees": 32, "max_features": 0.796901, "bootstrap": True, "min_samples_split": 7}
SCHWEFEL_HPO = {"number_of_trees": 4, "max_features": 0.134579, "bootstrap": False, "min_samples_split": 7}

ITERATIONS = 2

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
def run_test(function_name, function, global_minimum, values, optimization_objectives, hpo, run_no):
    dim = len(values)
    GP_dir = "GPvsRF/" + function_name + "/GP/"
    RF_dir = "GPvsRF/" + function_name + "/RF/"
    if not os.path.exists(GP_dir): os.makedirs(GP_dir)
    if not os.path.exists(RF_dir): os.makedirs(RF_dir)

    GP_json = GP_dir + "GP_" + function_name + "_scenario.json"
    RF_json = RF_dir + "RF_" + function_name + "_scenario.json"
    GP_csv = GP_dir + "GP_output_samples_" + str(run_no) + ".csv"    
    RF_csv = RF_dir + "RF_output_samples_" + str(run_no) + ".csv"
    
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
    scenario["acquisition_function"] = "EI"
    scenario["output_data_file"] = GP_csv    

    with open(GP_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)

    scenario["application_name"] = "RF_" + function_name
    scenario["models"]["model"] = "random_forest"
    scenario["models"]["model"] = "random_forest"
    scenario["models"]["number_of_trees"] = hpo["number_of_trees"]
    scenario["models"]["max_features"] = hpo["max_features"]
    scenario["models"]["bootstrap"] = hpo["bootstrap"]
    scenario["models"]["min_samples_split"] = hpo["min_samples_split"]
    scenario["output_data_file"] = RF_csv

    with open(RF_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(GP_json, function)
    optimizer.optimize(RF_json, function)
    sys.stdout = stdout


# GET FUNCTIONS LIST where fi = [GP_dir, RF_dir, name, function, minimum, values, objective, n_iter, v_type, run_no]
def get_function_lists(dim):    
    functions = list()
    functions.append(["Ackley", ackley_function, [0] * (dim+1), [[-2, 2]] * dim, ["value"], ACKLEY_HPO, 0])
    functions.append(["Griewank", griewank_function, [0] * (dim+1), [[-2, 2]] * dim, ["value"], GRIEWANK_HPO, 0])
    functions.append(["Rastrigin", rastrigin_function, [0] * (dim+1), [[-2, 2]] * dim, ["value"], RASTRIGIN_HPO, 0])
    functions.append(["Schwefel", schwefel_function, [*list([420.9687]*dim), 0], [[0, 500]] * dim, ["value"], SCHWEFEL_HPO, 0])
    return functions


# PLOT OPTIMIZATION RESULTS
def plot_optimization(function_name, dim):
    # create file sh for plotting optimization results
    f = open("plot_optimization.sh", "w+")
    f.write('hm-plot-optimization-results -j $1 -i $2 $3 -o $4 -t "$5" -l "$6" "$7"')
    f.close()
    # create directory if not exists
    output_dir ="GPvsRF/" + function_name +  "/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # run sh
    file_dir = output_dir + "GP/GP_" + function_name + "_scenario.json"
    output_file = output_dir + "optimization_results_dim" + str(dim) + ".pdf"
    subprocess.Popen(["bash", "plot_optimization.sh", file_dir, output_dir + "GP/", output_dir + "RF/", output_file, r'\rm{Regret ' + function_name + "}", r'\rm{GP}', r'\rm{RF}'])


# MAIN
def main(dim, n_executions):
    # get function list
    functions = get_function_lists(dim)

    # for each function, run tests and plot optimization results
    for f in functions:
        for _ in range(n_executions):
            run_test(f[0], f[1], f[2], f[3], f[4], f[5], f[6])
            f[6] += 1 
        plot_optimization(f[0], dim)


# dimension, executions
main(2, 2)
#main(12, 200, 5, "real")