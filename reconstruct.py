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
logger = logging.getLogger(__name__)


# 1. param dict

params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels","allow_cycles","labeldistribution"],
    'allow_cycles':[False], # cycles are very bad
    'number_of_graphs' : [210,1010],
    'size_of_graphs' :[8] ,
    'node_labels' : [2,4,8],
    'edge_labels' : [2,4], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] # real is unnecessary
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
                                     **args) for args in tasklist], ".tasks")







# call with reconstruct.py TASKID  REPEATID


params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [10,20], # seems to help a little with larger problems, >3 recommended
    'n_neighbors' :[20,30] # seems to not matter much 25 and 50 look the same, 15 and 75 also
}
instancemakerparams = [{"n_landmarks":10, "n_neighbors":15},{ "n_landmarks":20, "n_neighbors":25} ] 
#maketasks(params_insta)

params_opt = {
    'keyorder' :  ["half_step_distance",'n_iter','multiproc',"add_grammar_rules","keeptop"],
    "half_step_distance" : [True], # true clearly supperior
    "n_iter":[10,15], # 5 just for ez problems
    "keeptop":[20], # 20 seems enough
    'multiproc': [False],
    "add_grammar_rules":[True]
}

Optimizerparams = maketasks(params_opt)


def reconstruct_and_evaluate(target_graph,
                                landmark_graphs,
                                desired_distances,
                                ranked_graphs,
                                **args):
    optimizer = pareto.LocalLandmarksDistanceOptimizer(**args)
    # providing target, prints real distances for all "passing" creations
    res = optimizer.optimize(landmark_graphs,desired_distances,ranked_graphs) #,target=target_graph)
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


    task_id = args[0] # 16    the graph configurations
    graphs = task [task_id]

    im_param_id= args[1]# 4    landmark graphs , n neighs
    im_params = instancemakerparams[im_param_id]


    optimizer_para_id = args[2]# 4  optimizer args,, e.g. n_iter halfstep dist
    optimizerargs = Optimizerparams[optimizer_para_id]

    logger.debug(im_params)
    logger.debug(tasklist[task_id])
    logger.debug(optimizerargs)

    run_id =args[3] # 10

    im =  InstanceMaker(**im_params).fit(graphs, 10)

    res = im.get(run_id)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = res
    result = reconstruct_and_evaluate( target_graph,
            landmark_graphs,
            desired_distances,
            ranked_graphs,
            **optimizerargs)


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




def imtostr(im):
    d=instancemakerparams[im]
    return "marks:%d neigh:%d" % (d["n_landmarks"], d["n_neighbors"])
def optitostr(op):
    d=Optimizerparams[op]
    return "top:%d iter:%d" % (d["keeptop"], d["n_iter"])
def grtostr(gr):
    d = tasklist[gr]
    #return "Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])
    #return tuple(("Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])).split(" "))
    return tuple(("numgr:%d elab:%d nlab:%d siz:%d" % (d['number_of_graphs'],d['edge_labels'],d['node_labels'],d['size_of_graphs'])).split(" "))

def report():
    dat= defaultdict(dict)

    for a in range(len(tasklist)):
        for b in range(len(instancemakerparams)):
            for c in range(len(Optimizerparams)):
                dat[(imtostr(b),optitostr(c))][grtostr(a)] = getvalue(a,b,c)

    import pprint
    print (pandas.DataFrame(dat).to_string())
    #print (pandas.DataFrame(dat))df.describe().to_string()
    '''
    print ("instancemaker params:")
    pprint.pprint(instancemakerparams) 
    print ("optimizer params:")
    pprint.pprint(Optimizerparams) 
    print ("graph configurations:")
    pprint.pprint(tasklist  )

    '''





