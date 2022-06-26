import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from hypermapper import optimizer
stdout = sys.stdout 

ACKLEY_RANGE =          [-2, 2]
GRIEWANK_RANGE =        [-5, 5]
RASTRIGIN_RANGE =       [-2, 2]
SCHWEFEL_RANGE =        [-500, 500]
DROPWAVE_RANGE =        [-1, 1]
EASOM_RANGE =           [-5, 5]
MICHALEWICZ_RANGE =     [-4, 4]
STYBLINSKI_TANG_RANGE = [-8, 8]


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

# PLOT 2d FUNCTION
def plot_graph(function_name, function, global_minimum, values, resolution=50, dpi=100):
    directory = function_name + "/"
    if not os.path.exists(directory): os.makedirs(directory)

    # plt detail
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = plt.axes(projection='3d')
    ax.view_init(30, -45) 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.grid(lw=0.5)

    # define X, Y and Z axes
    X, Y = np.meshgrid(np.linspace(values[0][0], values[0][1], resolution), np.linspace(values[1][0], values[1][1], resolution))
    Z = function({"x1" : X, "x2": Y})   

    # plot objective function
    ax.contour(X, Y, Z, 50, cmap='viridis', alpha=0.50)
    ax.set_xlabel(r'\rm{\Large{{x1}}')
    ax.set_ylabel(r'\rm{\Large{{x2}}')
    ax.set_zlabel(r'\rm{\Large{{value}}')

    # plot actual minimum of objective function
    ax.scatter(global_minimum[0], global_minimum[1], global_minimum[2], color='red', label=r'\rm{\large{Global Minimum}}', alpha=0.40)
    
    descr = r'\rm{\Large{global minimum: \(f(\bf{x\ast}) = ' + str(global_minimum[2]) + r'\), at \(\bf{x\ast} = (' + str(global_minimum[0]) + ', ' + str(global_minimum[1]) + r')\)}}'
    
    ax.legend()
    plt.figtext(0.5, 0.01, descr, wrap=True, horizontalalignment='center', fontsize=12)
    plt.title(r'\rm{\huge{' + function_name + '}}')
    plt.savefig(directory + 'fig.pdf', bbox_inches='tight')  


# GET FUNCTIONS LIST where fi = [GP_dir, RF_dir, name, function, minimum, values, objective, n_iter, v_type, run_no]
def get_function_lists(dim=2):    
    function_info = list()
    function_info.append(["Drop-Wave", dropwave_function_2d, [0, 0, -1], [DROPWAVE_RANGE] * dim, ["value"]])
    function_info.append(["Easom", easom_function_2d, [3.14, 3.14, -1], [EASOM_RANGE] * dim, ["value"]])
    function_info.append(["Michalewicz", michalewicz_function_2d, [2.20, 1.57, -1.8013], [MICHALEWICZ_RANGE] * dim, ["value"]])
    function_info.append(["Styblinski-Tang", styblinski_tang_function_2d, [-2.903, -2.903, -39.166], [STYBLINSKI_TANG_RANGE] * dim, ["value"]])
    function_info.append(["Ackley", ackley_function_2d, [0] * (dim+1), [ACKLEY_RANGE] * dim, ["value"]])
    function_info.append(["Griewank", griewank_function_2d, [0] * (dim+1), [GRIEWANK_RANGE] * dim, ["value"]])
    function_info.append(["Rastrigin", rastrigin_function_2d, [0] * (dim+1), [RASTRIGIN_RANGE] * dim, ["value"]])
    function_info.append(["Schwefel", schwefel_function_2d, [*list([420.9687]*dim), 0], [SCHWEFEL_RANGE] * dim, ["value"]])    
    return function_info

# MAIN
def main():
    # get function list
    functions = get_function_lists()

    # for each function, run tests and plot optimization results
    for f in functions:
        plot_graph(f[0], f[1], f[2], f[3])


main()