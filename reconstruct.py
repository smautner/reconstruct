from util import random_graphs as rg
from collections import defaultdict
import pandas
import os
import sys
import numpy as np

from util.util import jdumpfile, jloadfile, InstanceMaker, dumpfile, loadfile

from util import rule_rand_graphs as rrg
EXPERIMENT_REPEATS = 20

'''
USAGE:
    python3 reconstruct.py  to generate problem instances
    fish runall.sh  to run with parallel
    python3 -c "import reconstruct as r; r.report()"   to see result
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
    'number_of_graphs': [500,2000],
    'size_of_graphs' :[8] ,
    'node_labels' : [4,8],
    'edge_labels' : [2,4], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] ,# real is unnecessary
    'maxdeg':[3],
    # rule rand graphs , iter argument ,  
    #0 means just use the rand graphs, a little hacky but works for now
    'rrg_iter':[0]
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
def make_task_file():
    def maketsk(args):
        rrg_iter = args.pop("rrg_iter")
        graphs = rg.make_graphs_static(**args)
        if rrg_iter > 0:
            graphs = rrg.rule_rand_graphs(graphs, numgr=500,iter=rrg_iter) 
        return graphs
    dumpfile([maketsk(args) for args in tasklist], ".tasks")
    #dumpfile([ rg.make_graphs_static(maxdeg=3, **args) for args in tasklist], ".tasks")


def load_chem(AID):
    import json
    import networkx.readwrite.json_graph as sg
    with open(AID+".json", 'r') as handle:
        js = json.load(handle.read())
        res = [sg.node_link_graph(jsg) for jsg in js]
    for g in res:
        g.graph={}
    return res



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
    'keyorder' :  ["half_step_distance",'n_iter','multiproc',"add_grammar_rules","keeptop","squared_error","graph_size_limiter"],
    "half_step_distance" : [True], # true clearly supperior
    "n_iter":[15,20], # 5 just for ez problems
    "keeptop":[10,15], # 20 seems enough
    'multiproc': [10],
    "add_grammar_rules":[True],
    "squared_error": [False], # False slightly better 590:572 
    "graph_size_limiter":[ lambda x: x.max()+(int(x.std()) or 5) ]
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
        params_args = {"keyorder":[0,1,2,3],
                        0:range(len(tasklist)),
                        1:range(len(instancemakerparams)),
                        2:range(len(Optimizerparams)),
                        3:range(EXPERIMENT_REPEATS),
                        }
        args = maketasks(params_args)
        args=args[arg]


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

    run_id =args[3] 
    im =  InstanceMaker(**im_params).fit(graphs, EXPERIMENT_REPEATS)
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




def getvalue(a,b,c, nores, nosucc): # nosucc and nores are just collecting stats
    completed = 0
    allsteps=[-1]
    success = 0
    for task in range(EXPERIMENT_REPEATS):
        taskname = "%d_%d_%d_%d" % (a,b,c,task)
        fname = ".res/"+taskname
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




#####################
# Output formatters... 
#######################
def defaultformatter(paramsdict, instance):
    res =[]
    for k in paramsdict['keyorder']:
        if len(paramsdict[k] )> 1:
            #  interesting key
            res.append("%s:%d " % ( k[:4],instance[k]) )
    return tuple(res)

def imtostr(im):
    d=instancemakerparams[im]
    return "marks:%d neigh:%d" % (d["n_landmarks"], d["n_neighbors"])

def optitostr(op):
    d=Optimizerparams[op]
    return defaultformatter(params_opt,d)
    #return "top:%d iter:%d" % (d["keeptop"], d["n_iter"])
    #return "grsizelimit:%d"  % (d["graph_size_limiter"])

def grtostr(gr):
    d = tasklist[gr]
    return defaultformatter(params_graphs,d)
    #return "Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])
    #return tuple(("Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])).split(" "))
    #return tuple(("elab:%d nlab:%d" % (d['edge_labels'],d['node_labels'])).split(" "))
    #return tuple(("elab:%d nlab:%d graphs:%d rrg_it:%d" % (d['edge_labels'],d['node_labels'],d['number_of_graphs'],d['rrg_iter'])).split(" "))

def report():
    dat= defaultdict(dict)
    nores = []
    nosucc =[]
    for a in range(len(tasklist)):
        for b in range(len(instancemakerparams)):
            for c in range(len(Optimizerparams)):
                #dat[(imtostr(b),optitostr(c))][grtostr(a)]= getvalue(a,b,c, nores, nosucc)
                dat[optitostr(c)][grtostr(a)]= getvalue(a,b,c, nores, nosucc)

    import pprint
    print (pandas.DataFrame(dat).to_string())
    mod = lambda x : str(x).replace("_",' ')
    print ("nores",mod(nores))
    print ('nosucc',mod(nosucc))
    print ("sumsuccess:", sum([int(a) for c in dat.values() for a,b in c.values()]))
    print ("maxrnd:", max([int(b) for c in dat.values() for a,b in c.values()]))

