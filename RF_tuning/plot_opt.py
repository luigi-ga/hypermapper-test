import os
import subprocess

def plot_optimization(json_dir, dirs, output_file, f_name, folders):
    code = 'hm-plot-optimization-results -j ' + json_dir + ' -i '
    for folder in folders:
        code += dirs + str(folder) + " " 
    code += '-o ' + dirs + output_file + ' -t "$1" -l '
    for folder in folders:
        code += '"\\rm{' + folder + '}" '

    # create file sh for plotting optimization results
    f = open("plot_optimization.sh", "w+")
    f.write(code)
    f.close()
    subprocess.Popen(["bash", "plot_optimization.sh", r'\rm{' + f_name + ' Regret}'])


#plot_optimization("Tests/Ackley/test0/Ackley_scenario.json", "Tests/Ackley/", "top5_regret.pdf", "Ackley", ["test33", "test2", "test18", "test0"])
#plot_optimization("Tests/Griewank/test0/Griewank_scenario.json", "Tests/Griewank/", "top5_regret.pdf", "Griewank", ["test36", "test4", "test39", "test23"])
#plot_optimization("Tests/Rastrigin/test0/Rastrigin_scenario.json", "Tests/Rastrigin/", "top5_regret.pdf", "Rastrigin", ["test11", "test40", "test31", "test33"])
#plot_optimization("Tests/Schwefel/test0/Schwefel_scenario.json", "Tests/Schwefel/", "top5_regret.pdf", "Schwefel", ["test19", "test25", "test4", "test26"])

#plot_optimization("Tests/Ackley/test0/Ackley_scenario.json", "Tests/Ackley/", "progress_regret.pdf", "Ackley", ["test33", "test26", "test34", "test15"])
#plot_optimization("Tests/Griewank/test0/Griewank_scenario.json", "Tests/Griewank/", "progress_regret.pdf", "Griewank", ["test36", "test29", "test5", "test1"])
#plot_optimization("Tests/Rastrigin/test0/Rastrigin_scenario.json", "Tests/Rastrigin/", "progress_regret.pdf", "Rastrigin", ["test11", "test19", "test14", "test13"])
#plot_optimization("Tests/Schwefel/test0/Schwefel_scenario.json", "Tests/Schwefel/", "progress_regret.pdf", "Schwefel", ["test19", "test9", "test44", "test0"])

