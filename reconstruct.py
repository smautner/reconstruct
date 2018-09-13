from util import random_graphs as rg
from collections import defaultdict
import pandas
import os
import sys

from util.util import jdumpfile, jloadfile, InstanceMaker, dumpfile, loadfile

#echo -e (seq 0 15)" "(seq 0 3)" "(seq 0 1)" "(seq 0 9)"\n" | parallel --bar --results .log -j 15 python3 reconstruct.py 
####################
# run and get best result
###################
from exploration import pareto


#echo -e  (seq 10)" "(seq 10)"\n" | parallel                09-13-1154




from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)




# 1. param dict

params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels"],
    'number_of_graphs' : [110,510],
    'size_of_graphs' :[5,10] ,
    'node_labels' : [5,10],
    'edge_labels' : [5,10]
}

# 2. function paramdict to tasks

def maketasks(params):
    # want a list
    combolist =[[]]
    for key in params['keyorder']:
        combolist = [  e+[value] for value in params[key]  for e in combolist ]

    return  [ {k:v for k,v in zip(params['keyorder'],configuration)} for configuration in combolist ]

tasklist  = maketasks(params_graphs )


# 3. loop over tasks

def make_task_file():
    dumpfile([ rg.make_graphs_static(maxdeg=3,
                                     allow_cycles=False,
                                     labeldistribution='uniform',
                                     **args) for args in tasklist], ".tasks")







# call with reconstruct.py TASKID  REPEATID


params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [5,10],
    'n_neighbors' :[25,50]
}
instancemakerparams = maketasks(params_insta)

params_opt = {
    'keyorder' :  ["half_step_distance"],
    "half_step_distance" : [True, False]
}

Optimizerparams = maketasks(params_opt)


def reconstruct_and_evaluate(target_graph,
                                landmark_graphs,
                                desired_distances,
                                ranked_graphs,
                                **args):
    optimizer = pareto.LocalLandmarksDistanceOptimizer(**args)
    res = optimizer.optimize(landmark_graphs,desired_distances,ranked_graphs)
    return res


if __name__=="__main__":

    if len(sys.argv)==1:
        print("writing task file...")
        make_task_file()
        exit()
    else:
        args = list(map(int, sys.argv[-1].strip().split(" ")))


    resu=[]
    task = loadfile(".tasks")


    task_id = args[0] # 16
    graphs = task [task_id]

    im_param_id= args[1]# 4
    im_params = instancemakerparams[im_param_id]


    optimizer_para_id = args[2]# 2
    optimizerargs = Optimizerparams[optimizer_para_id]

    run_id =args[3] # 10

    im =  InstanceMaker(**im_params).fit(graphs, 10)

    res = im.get(run_id)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = res
    result = reconstruct_and_evaluate( target_graph, landmark_graphs, desired_distances, ranked_graphs, **optimizerargs)


    dumpfile(result, ".res/%d_%d_%d_%d" % (task_id, im_param_id, optimizer_para_id, run_id))   #!!!



#######################################
# Report
#########################




def getvalue(a,b,c):
    completed = 0
    success = 0
    for task in range(10):
        fname = ".res/%d_%d_%d_%d" % (a,b,c,task)
        if os.path.isfile(fname):
            completed +=1
            success += loadfile(fname)
    return success, completed

def report():
    dat= defaultdict(dict)

    for a in range(len(tasklist)):
        for b in range(len(instancemakerparams)):
            for c in range(len(Optimizerparams)):
                dat[a][(b,c)] = getvalue(a,b,c,d)
    return pandas.DataFrame(dat)








