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
NUM_COMPARISON = 4   # regret comparison in generated pdf

TIME_BUDGET_GP = 2800
TIME_BUDGET_GPtune = 7
MAX_ITERATIONS_GP = 1
MAX_ITERATIONS_GPtune = 1
DIM = 1
REPETITIONS = 2

ACKLEY_RANGE = [-5, 5]
GRIEWANK_RANGE = [-5, 5]
RASTRIGIN_RANGE = [-5, 5]
SCHWEFEL_RANGE = [420.9687-5, 420.9687+5]

run_no = 0

AF = ["UCB", "TS", "EI"] 


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
def gaussian_process(hyperparameters):
    global run_no
    min = 0
    global run_no
    min = 0
    for f in FUNCTIONS:
        minf = 0
        for i in range(REPETITIONS): minf += gaussian_process_optimizer(f[0], f[1], f[2], f[4], hyperparameters, run_no, i)
        minf /= REPETITIONS
        write_description_file(f[0], hyperparameters, minf, minf/f[4], run_no) 
        min += (minf/f[4])
    run_no += 1    
    return min/len(FUNCTIONS)


def gaussian_process_optimizer(function_name, function, values, max, hyperparameters, run_no, rep_no):
    directory = "Tests/" + "dim" + str(DIM) + "/" + function_name + "/test" + str(run_no) + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    json_dir = directory + "scenario.json"
    csv_dir = directory + "output_samples_" + str(rep_no) + ".csv"

    scenario = {}    
    scenario["application_name"] = "GP_" + function_name
    scenario["models"] = {}
    scenario["optimization_objectives"] = ["value"]
    scenario["optimization_iterations"] = MAX_ITERATIONS_GPtune
    scenario["time_budget"] = TIME_BUDGET_GPtune
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
   
    scenario["models"]["model"] = "gaussian_process"      
    scenario["models"]["acquisition_function"] = AF[int(hyperparameters["acquisition_function"])]
    scenario["models"]["local_search_starting_points"] = pow(10, int(hyperparameters["local_search_starting_points"]))
    scenario["models"]["local_search_random_points"] = pow(10, int(hyperparameters["local_search_random_points"]))
    scenario["models"]["log_transform_output"] = bool(hyperparameters["log_transform_output"])
    scenario["models"]["lengthscale_prior"] = {}
    scenario["models"]["lengthscale_prior"]["name"] = "gamma"
    scenario["models"]["lengthscale_prior"]["parameters"] = [1.3, 0.1]
    scenario["models"]["epsilon_greedy_threshold"] = int(hyperparameters["epsilon_greedy_threshold"])
    scenario["models"]["predict_noiseless"] = bool(hyperparameters["predict_noiseless"]) 
    scenario["models"]["exploration_augmentation"] = pow(10, int(hyperparameters["exploration_augmentation"]))

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, function)

    # return the minimum value found by hypermapper
    df = pd.read_csv(csv_dir)
    min = float(df['value'].min())    
    return min
    

# run bayesyan optimization to tune gaussian process hyperparameters
def GP_hyperparameter_tuning(function):
    directory = "Optimizer/dim" + str(DIM) + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    csv_dir = directory + "optimizer_dim" + str(DIM) + ".csv"
    json_dir = directory + "optimizer_dim" + str(DIM) + ".json"

    scenario = {}    
    scenario["application_name"] = "GP optimizer"
    scenario["models"] = {}
    scenario["optimization_objectives"] = ["value"]
    scenario["optimization_iterations"] = MAX_ITERATIONS_GP
    scenario["time_budget"] = TIME_BUDGET_GP
    scenario["normalize_inputs"] = True
    scenario["output_data_file"] = csv_dir

    scenario["resume_optimization"] = RESUME
    scenario["resume_optimization_data"] = "/mnt/d/Users/Luigi/Desktop/Universita/Tirocinio/GP_tuning/Optimizer/dim" + str(DIM) + "/optimizer_dim" + str(DIM) + ".csv"
    
    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = 5   # d + 1

    scenario["input_parameters"] = {} 

    acquisition_function = {}
    acquisition_function["parameter_type"] = "integer"            
    acquisition_function["values"] = [0, 2]
    scenario["input_parameters"]["acquisition_function"] = acquisition_function
    
    local_search_starting_points = {}
    local_search_starting_points["parameter_type"] = "integer"             
    local_search_starting_points["values"] = [0, 1]   
    scenario["input_parameters"]["local_search_starting_points"] = local_search_starting_points

    local_search_random_points = {}
    local_search_random_points["parameter_type"] = "integer"               
    local_search_random_points["values"] = [0, 3]  
    scenario["input_parameters"]["local_search_random_points"] = local_search_random_points      

    log_transform_output = {}
    log_transform_output["parameter_type"] = "integer"            
    log_transform_output["values"] = [0, 1]
    scenario["input_parameters"]["log_transform_output"] = log_transform_output

    lengthscale_prior = {}
    lengthscale_prior["parameter_type"] = "integer"            
    lengthscale_prior["values"] = [0, 1]
    scenario["input_parameters"]["lengthscale_prior"] = lengthscale_prior   

    epsilon_greedy_threshold = {}
    epsilon_greedy_threshold["parameter_type"] = "integer"            
    epsilon_greedy_threshold["values"] = [0, 1]
    scenario["input_parameters"]["epsilon_greedy_threshold"] = epsilon_greedy_threshold

    predict_noiseless = {}
    predict_noiseless["parameter_type"] = "integer"            
    predict_noiseless["values"] = [0, 1]
    scenario["input_parameters"]["predict_noiseless"] = predict_noiseless

    exploration_augmentation = {}
    exploration_augmentation["parameter_type"] = "integer"            
    exploration_augmentation["values"] = [-5, -2]
    scenario["input_parameters"]["exploration_augmentation"] = exploration_augmentation 

    scenario["models"]["model"] = "gaussian_process"
    scenario["acquisition_function"] = "EI"

    with open(json_dir, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(json_dir, function)
    sys.stdout = stdout


# write a file with test info in test directory
def write_description_file(f_name, hyperparameters, min, norm_min, run_no):
    csv_header = ["test_number"]
    for hp in hyperparameters: csv_header.append(hp)
    csv_header += ["mean_min", "mean_normalized_min"]

    csv_row = [ run_no,  
                AF[int(hyperparameters["acquisition_function"])],
                pow(10, int(hyperparameters["local_search_starting_points"])),
                pow(10, int(hyperparameters["local_search_random_points"])),
                bool(hyperparameters["log_transform_output"]),
                "gamma",
                int(hyperparameters["epsilon_greedy_threshold"]),
                bool(hyperparameters["predict_noiseless"]),
                pow(10, int(hyperparameters["exploration_augmentation"])),
                min, 
                norm_min
              ]

    dir = "Tests/" + "dim" + str(DIM) + "/" + f_name + "/tests_specs.csv"
    with open(dir, "a+", newline='') as write_obj:
        if run_no == 0: csv.writer(write_obj).writerow(csv_header)
        csv.writer(write_obj).writerow(csv_row)



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


GP_hyperparameter_tuning(gaussian_process)