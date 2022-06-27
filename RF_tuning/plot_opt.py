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


#plot_optimization("Tests/Ackley/test0/Ackley_scenario.json", "Tests/Ackley/", "top5_regret.pdf", "Ackley", ["test44", "test14", "test53", "test4", "test7"])
#plot_optimization("Tests/Griewank/test0/Griewank_scenario.json", "Tests/Griewank/", "top5_regret.pdf", "Griewank", ["test23", "test51", "test49", "test30", "test18"])
#plot_optimization("Tests/Rastrigin/test0/Rastrigin_scenario.json", "Tests/Rastrigin/", "top5_regret.pdf", "Rastrigin", ["test17", "test49", "test41", "test12", "test43"])
#plot_optimization("Tests/Schwefel/test0/Schwefel_scenario.json", "Tests/Schwefel/", "top5_regret.pdf", "Schwefel", ["test9", "test20", "test19", "test35", "test2"])

#plot_optimization("Tests/Ackley/test0/Ackley_scenario.json", "Tests/Ackley/", "progress_regret.pdf", "Ackley", ["test8", "test41", "test45", "test0", "test37", "test44"])
#plot_optimization("Tests/Griewank/test0/Griewank_scenario.json", "Tests/Griewank/", "progress_regret.pdf", "Griewank", ["test47", "test2", "test53", "test32", "test17", "test23"])
#plot_optimization("Tests/Rastrigin/test0/Rastrigin_scenario.json", "Tests/Rastrigin/", "progress_regret.pdf", "Rastrigin", ["test51", "test7", "test13", "test26", "test0", "test17"])
plot_optimization("Tests/Schwefel/test0/Schwefel_scenario.json", "Tests/Schwefel/", "progress_regret.pdf", "Schwefel", ["test4", "test48", "test54", "test5", "test46", "test9"])

