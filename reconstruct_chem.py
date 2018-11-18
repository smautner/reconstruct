from util import random_graphs as rg
from collections import defaultdict
import pandas
import os
import sys
import numpy as np





######################################
### THIS IS THE MEGA HACK FOR CHEM GRAPHS
#####################################












from util.util import jdumpfile, jloadfile, InstanceMaker, dumpfile, loadfile

from util import rule_rand_graphs as rrg
EXPERIMENT_REPEATS = 20

'''
USAGE:
    python3 reconstruct.py  to generate problem instances
    fish runall.sh  to run with parallel
    python3 -c "import reconstruct_chem as r; r.report()"   to see result
'''


####################
# run and get best result
###################
from exploration import pareto


from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)
logger = logging.getLogger(__name__)



#############################################
##  OPTIONS FOR GRAPHS
##########################################

# 1. param dict

params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels","allow_cycles","labeldistribution","maxdeg","rrg_iter"],
    'allow_cycles':[False], # cycles are very bad
    'number_of_graphs': [20,40],
    'size_of_graphs' :[8] ,
    'node_labels' : [4,8],
    'edge_labels' : [2,4,6], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] ,# real is unnecessary
    'maxdeg':[3],
    'rrg_iter':[2,3,4]# rule rand graphs , iter argument
}

# 2. function paramdict to tasks

def maketasks(params):
    # want a list
    combolist =[[]]
    for key in params['keyorder']:
        combolist = [  e+[value] for value in params[key]  for e in combolist ]

    return  [ {k:v for k,v in zip(params['keyorder'],configuration)} for configuration in combolist ]

tasklist  = maketasks(params_graphs )


# 3. loop over task

import json
import random
import networkx as nx
from networkx.readwrite.json_graph import node_link_graph
def make_task_file():

    with open("AID1835Active.json","r") as f:
            stuff = f.read()
    s1 = json.loads(stuff)
    s1= [node_link_graph(s) for s in s1]
    
    for g in s1:
        g.graph = {}

    random.shuffle(s1)
    dumpfile(s1 , ".chemtask")

    from structout import gprint
    g=s1[0]
    print (g.__dict__)
    pos=nx.drawing.nx_agraph.graphviz_layout(g, prog='neato', args="-Gratio='2'")

    gprint(g)
    #dumpfile([ rg.make_graphs_static(maxdeg=3, **args) for args in tasklist], ".tasks")





######################
#  OPTIONS FOR PROBLEM GENERATOR
##########################

# call with reconstruct.py TASKID  REPEATID


params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [25], # seems to help a little with larger problems, >3 recommended
    'n_neighbors' :[50] # seems to not matter much 25 and 50 look the same, 15 and 75 also
    }
instancemakerparams =maketasks(params_insta)

############################
#  OPTIONS FOR SOLVER 
##############################

params_opt = {
    'keyorder' :  ["half_step_distance",'n_iter','multiproc',"add_grammar_rules","keeptop","graph_size_limiter"],
    "half_step_distance" : [True], # true clearly supperior
    "n_iter":[20], # 5 just for ez problems
    "keeptop":[10], # 20 seems enough
    'multiproc': [False],
    "add_grammar_rules":[True],
    "graph_size_limiter":[1]
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
        #print(sys.argv[-1])
        #args = list(map(int, sys.argv[-1].strip().split(" ")))
        
        # ok need to run this on the cluster where i only have a task id...
        # the queickest way to hack this while still being compatible with the old crap 
        # is using the maketasts function defined above...
        arg = int(sys.argv[-1])-1
        params_args = {"keyorder":[0,1,2],
                        0:range(len(instancemakerparams)),
                        1:range(len(Optimizerparams)),
                        2:range(EXPERIMENT_REPEATS),
                        }
        args = maketasks(params_args)
        args=args[arg]


        
    resu=[]
    graphs = loadfile(".chemtask")
    print (graphs)

    im_param_id= args[0]# 4    landmark graphs , n neighs
    im_params = instancemakerparams[im_param_id]


    optimizer_para_id = args[1]# 4  optimizer args,, e.g. n_iter halfstep dist
    optimizerargs = Optimizerparams[optimizer_para_id]

    logger.debug(im_params)
    logger.debug(optimizerargs)

    run_id =args[2] 

    im =  InstanceMaker(**im_params).fit(graphs, EXPERIMENT_REPEATS)

    res = im.get(run_id)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = res
    result = reconstruct_and_evaluate( target_graph,
            landmark_graphs,
            desired_distances,
            ranked_graphs,
            **optimizerargs)


    dumpfile(result, ".chemres/%d_%d_%d" % (im_param_id, optimizer_para_id, run_id))   #!!!



#######################################
# Report
#########################




def getvalue(a,b, nores, nosucc): # nosucc and nores are just collecting stats
    completed = 0
    allsteps=[-1]
    success = 0
    for task in range(EXPERIMENT_REPEATS):
        taskname = "%d_%d_%d" % (a,b,task)
        fname = ".chemres/"+taskname
        if os.path.isfile(fname):
            completed +=1
            res, steps = loadfile(fname)
            success += res
            if not res:   # FAIL
                nosucc.append(taskname)
            else:       # success -> remember step count
                allsteps.append(steps)
        else: 
            nores.append(taskname)
    allsteps = np.array(allsteps)
    return success,  allsteps.max()






def imtostr(im):
    d=instancemakerparams[im]
    return "marks:%d neigh:%d" % (d["n_landmarks"], d["n_neighbors"])
#####################
# formatting for solver options 
#######################
def optitostr(op):
    d=Optimizerparams[op]
    return "top:%d iter:%d" % (d["keeptop"], d["n_iter"])
    #return "grsizelimit:%d"  % (d["graph_size_limiter"])

###################
# formatting of y axis -- the graphs
##############################
def grtostr(gr):
    d = tasklist[gr]
    #return "Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])
    #return tuple(("Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])).split(" "))
    #return tuple(("elab:%d nlab:%d" % (d['edge_labels'],d['node_labels'])).split(" "))
    return tuple(("elab:%d nlab:%d graphs:%d rrg_it:%d" % (d['edge_labels'],d['node_labels'],d['number_of_graphs'],d['rrg_iter'])).split(" "))

def report():
    dat= defaultdict(dict)
    nores = []
    nosucc =[]
    for a in range(len(instancemakerparams)):
        for b in range(len(Optimizerparams)):
            #dat[(imtostr(b),optitostr(c))][grtostr(a)]= getvalue(a,b,c, nores, nosucc)
            dat[imtostr(a)][optitostr(b)]= getvalue(a,b, nores, nosucc)

    import pprint
    print (pandas.DataFrame(dat).to_string())
    mod = lambda x : str(x).replace("_",' ')
    print ("nores",mod(nores))
    print ('nosucc',mod(nosucc))
    #print (pandas.DataFrame(dat))df.describe().to_string()
    '''
    print ("instancemaker params:")
    pprint.pprint(instancemakerparams) 
    print ("optimizer params:")
    pprint.pprint(Optimizerparams) 
    print ("graph configurations:")
    pprint.pprint(tasklist  )

    '''





