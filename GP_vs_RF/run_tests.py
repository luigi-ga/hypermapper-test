import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hypermapper import optimizer
stdout = sys.stdout 

ACKLEY_RANGE = [-5, 5]
GRIEWANK_RANGE = [-5, 5]
RASTRIGIN_RANGE = [-5, 5]
SCHWEFEL_RANGE = [421-5, 421+5]

# 2d FUNCTIONS

def ackley_function_2d(X):
    x1 = X["x1"]
    x2 = X["x2"]
    a, b, c = 20, 0.2, 2 * np.pi
    return -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2))) + a + np.exp(1)

def griewank_function_2d(X):
    x1 = X["x1"]
    x2 = X["x2"]
    return 1 + (x1**2 + x2**2) / 4000 - (np.cos(x1 / np.sqrt(1))) * (np.cos(x2 / np.sqrt(2)))

def rastrigin_function_2d(X):
    x1 = X["x1"]
    x2 = X["x2"]
    return 20 + (x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2))

def schwefel_function_2d(X):
    x1= X["x1"]
    x2 = X["x2"]
    return 2 * 418.9829 - (x1 * np.sin(np.sqrt(np.abs(x1))) + x2 * np.sin(np.sqrt(np.abs(x2))))

def styblinski_tang_function_2d(X):
    x1= X["x1"]
    x2 = X["x2"]
    return 0.5 * (x1**4 - 16 * x1**2 + 5 * x1 + x2**4 - 16 * x2**2 + 5 * x2)

def dropwave_function_2d(X):
    x1= X["x1"]
    x2 = X["x2"]
    return -(1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2)

def easom_function_2d(X):
    x1= X["x1"]
    x2 = X["x2"]
    return - np.cos(x1) * np.cos(x2) * np.exp(- (x1 - np.pi)**2 - (x2 - np.pi)**2)

def michalewicz_function_2d(X):
    x1= X["x1"]
    x2 = X["x2"]
    return - (np.sin(x1) * np.sin(x1**2 / np.pi)**2 + np.sin(x2) * np.sin(2 * x2**2 / np.pi)**2)


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


# PLOT 2d FUNCTION
def plotHM(f_info):   
    json_f = f_info[0]
    csv_f = f_info[1]
    function_name = f_info[2]
    function = f_info[3]
    global_minimum = f_info[4]
    output_dir = f_info[5]
    resolution = f_info[6]
    dpi = f_info[7]
    model = f_info[8]

    output_file_name = output_dir + model + "_" + function_name

    # open json file and read input parameters values
    f = open(json_f)
    scenario = json.load(f)
    x1_val = scenario["input_parameters"]["x1"]["values"]
    x2_val = scenario["input_parameters"]["x2"]["values"]
    init_iter = 10
    n_iter = scenario["optimization_iterations"] + init_iter
    f.close()

    # plt detail
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = plt.axes(projection='3d')
    ax.view_init(30, -45) 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.grid(lw=0.5)

    # define X, Y and Z axes
    X, Y = np.meshgrid(np.linspace(x1_val[0], x1_val[1], resolution), np.linspace(x2_val[0], x2_val[1], resolution))
    Z = function({"x1" : X, "x2": Y})   

    # plot objective function
    ax.contour(X, Y, Z, 50, cmap='viridis', alpha=0.50)
    ax.set_xlabel(r'\rm{\Large{x1}}')
    ax.set_ylabel(r'\rm{\Large{x2}}')
    ax.set_zlabel(r'\rm{\Large{value}}')

    # plot actual minimum of objective function
    ax.scatter(global_minimum[0], global_minimum[1], global_minimum[2], color='red', label=r'\rm{\large{Global Minimum}}', alpha=0.40)

    # open csv and plot points that hypermapper explored
    hm_output = pd.read_csv(csv_f)
    minimum = hm_output.iloc[[hm_output['value'].idxmin()]]
    color = "purple"
    for index, row in hm_output.iterrows():
        if float(row['value']) == float(minimum['value']):
            ax.scatter(row['x1'], row['x2'], row['value'], color="black", label=r'\rm{\large{Hypermapper Minimum}}', alpha=1, marker="*")            
        else:
            if index == init_iter:
                ax.scatter(row['x1'], row['x2'], row['value'], color=color, label=r'\rm{\large{Bayesian Optimization}}', alpha=alpha, marker=".")
            if index == init_iter - 1:
                ax.scatter(row['x1'], row['x2'], row['value'], color=color, label=r'\rm{\large{Initial Random Sampling}}', alpha=alpha, marker=".")
                color="blue"
            alpha = index / n_iter * 0.35 + 0.4
            ax.scatter(row['x1'], row['x2'], row['value'], color=color, alpha=alpha, marker=".")
     
    descr = r'\begin{center}\rm{\Large{\(f(\bf{x\star}) = ' + str(global_minimum[2]) + r'\), at \(\bf{x\star} = (' + str(global_minimum[0]) + ', ' + str(global_minimum[1]) + r')\) \\'
    descr += r'\(f(\bf{x}) =' + str(round(minimum.iloc[0]['value'], 4)) + r'\),  at \(\bf{x} = (' + str(round(minimum.iloc[0]['x1'], 4)) + ', ' + str(round(minimum.iloc[0]['x2'], 4)) + r')\)}}\end{center}'
    
    ax.legend()
    plt.title(r'\rm{\huge{' + function_name + " " + model + '}}')
    plt.figtext(0.5, 0.01, descr, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(output_file_name + '.pdf', bbox_inches='tight')  



# RUNNING TESTS
def run_test(f_info):
    GP_dir = f_info[0]
    RF_dir = f_info[1]
    function_name = f_info[2]
    function = f_info[3]
    global_minimum = f_info[4]
    values = f_info[5]
    optimization_objectives = f_info[6]
    iterations = f_info[7]
    v_type = f_info[8]    
    run_no = f_info[9]

    dim = len(values)

    GP_json = GP_dir + "GP_" + function_name + "_scenario.json"
    RF_json = RF_dir + "RF_" + function_name + "_scenario.json"
    GP_csv = GP_dir + "GP_" + function_name + "_output_samples_" + str(run_no) + ".csv"    
    RF_csv = RF_dir + "RF_" + function_name + "_output_samples_" + str(run_no) + ".csv"
    
    scenario = {}    
    scenario["models"] = {}
    scenario["optimization_objectives"] = optimization_objectives
    scenario["optimization_iterations"] = iterations
    scenario["noise"] = True
    scenario["normalize_inputs"] = True
    scenario["design_of_experiment"] = {}
    scenario["design_of_experiment"]["number_of_samples"] = dim + 1

    scenario["input_parameters"] = {}
    for i in range(dim):
        x = {}
        x["parameter_type"] = v_type
        x["values"] = values[i]
        scenario["input_parameters"]["x" + str(i+1)] = x    

    scenario["application_name"] = "GP_" + function_name
    scenario["models"]["model"] = "gaussian_process"
    scenario["acquisition_function"] = "EI"
    scenario["output_data_file"] = GP_csv    
    scenario["output_pareto_file"] = GP_dir + "GP_output_pareto_dim" + str(dim) + ".csv"

    with open(GP_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)

    scenario["application_name"] = "RF_" + function_name
    scenario["models"]["model"] = "random_forest"
    scenario["output_data_file"] = RF_csv
    scenario["output_pareto_file"] = RF_dir + "RF_output_pareto_dim" + str(dim) + ".csv"

    with open(RF_json, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)
        
    optimizer.optimize(GP_json, function)
    optimizer.optimize(RF_json, function)
    sys.stdout = stdout

    if dim == 2 and len(optimization_objectives) == 1:
        output_dir = v_type + "/" + function_name + "/3D Graphs/"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plotHM([GP_json, GP_csv, function_name, function, global_minimum, output_dir, 50, 100, "GP"])
        plotHM([RF_json, RF_csv, function_name, function, global_minimum, output_dir, 50, 100, "RF"])



# GET FUNCTIONS LIST where fi = [GP_dir, RF_dir, name, function, minimum, values, objective, n_iter, v_type, run_no]
def get_function_lists(dim, n_iter, v_type):    
    function_info = list()

    if dim == 2:
        function_info.append(["Ackley", ackley_function_2d, [0] * (dim+1), [ACKLEY_RANGE] * dim, ["value"]])
        function_info.append(["Griewank", griewank_function_2d, [0] * (dim+1), [GRIEWANK_RANGE] * dim, ["value"]])
        function_info.append(["Rastrigin", rastrigin_function_2d, [0] * (dim+1), [RASTRIGIN_RANGE] * dim, ["value"]])
        function_info.append(["Schwefel", schwefel_function_2d, [*list([420.9687]*dim), 0], [SCHWEFEL_RANGE] * dim, ["value"]])
        #function_info.append(["Drop-Wave", dropwave_function_2d, [0, 0, -1], [[-1, 1]] * dim, ["value"]])
        #function_info.append(["Easom", easom_function_2d, [3.14, 3.14, -1], [[-5, 5]] * dim, ["value"]])
        #function_info.append(["Michalewicz", michalewicz_function_2d, [2.20, 1.57, -1.8013], [[-4, 4]] * dim, ["value"]])
        #function_info.append(["Styblinski-Tang", styblinski_tang_function_2d, [-2.903, -2.903, -39.166], [[-8, 8]] * dim, ["value"]])
    else:
        function_info.append(["Ackley", ackley_function, [0] * (dim+1), [ACKLEY_RANGE] * dim, ["value"]])
        function_info.append(["Griewank", griewank_function, [0] * (dim+1), [GRIEWANK_RANGE] * dim, ["value"]])
        function_info.append(["Rastrigin", rastrigin_function, [0] * (dim+1), [RASTRIGIN_RANGE] * dim, ["value"]])
        function_info.append(["Schwefel", schwefel_function, [*list([420.9687]*dim), 0], [SCHWEFEL_RANGE] * dim, ["value"]])

    # list of functions
    functions = list()
    for f in function_info:
        GP_dir = v_type + "/" + f[0] + "/GP_" + f[0] + "/dim" + str(dim) + "/"
        RF_dir = v_type + "/" + f[0] + "/RF_" + f[0] + "/dim" + str(dim) + "/"             
        if not os.path.exists(GP_dir): os.makedirs(GP_dir)
        if not os.path.exists(RF_dir): os.makedirs(RF_dir)
        functions.append([GP_dir, RF_dir, f[0], f[1], f[2], f[3], f[4], n_iter, v_type, 0])
    
    return functions



# PLOT OPTIMIZATION RESULTS
def plot_optimization(f_info, v_type, dim):
    # create file sh for plotting optimization results
    f = open("plot_optimization.sh", "w+")
    f.write('hm-plot-optimization-results -j $1 -i $2 $3 -o $4 -t "$5" -l "$6" "$7"')
    f.close()
    # create directory if not exists
    output_dir = v_type + "/" + f_info[2] + "/Optimization_Results/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # run sh
    file_dir = f_info[0] + "GP_" + f_info[2] + "_scenario.json"
    output_file = output_dir + "optimization_results_dim" + str(dim) + ".pdf"
    subprocess.Popen(["bash", "plot_optimization.sh", file_dir, f_info[0], f_info[1], output_file, r'\rm{' + f_info[2] + " dim " + str(dim) + "}", r'\rm{GP}', r'\rm{RF}'])


# MAIN
def main(dim, n_iter, n_executions, v_type):
    # get function list
    functions = get_function_lists(dim, n_iter, v_type)

    # for each function, run tests and plot optimization results
    for f_info in functions:
        for _ in range(n_executions):
            run_test(f_info)
            f_info[9] += 1 
        plot_optimization(f_info, v_type, dim)


# dimension, iterations, executions, variables type

main(1, 1, 1, "real")