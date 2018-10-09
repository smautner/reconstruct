from util import random_graphs as rg
from collections import defaultdict
import pandas
import os
import sys

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


# 1. param dict

params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels","allow_cycles","labeldistribution","maxdeg","rrg_iter"],
    'allow_cycles':[False], # cycles are very bad
    'number_of_graphs' : [20, 30],
    'size_of_graphs' :[8] ,
    'node_labels' : [2,8],
    'edge_labels' : [2,4], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] ,# real is unnecessary
    'maxdeg':[3],
    'rrg_iter':[2,3]# rule rand graphs , iter argument
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
        g,_ = rrg.rule_rand_graphs(graphs, numgr=500,iter=rrg_iter) 
        return g #+ graphs

    dumpfile([maketsk(args) for args in tasklist], ".tasks")

    #dumpfile([ rg.make_graphs_static(maxdeg=3, **args) for args in tasklist], ".tasks")







# call with reconstruct.py TASKID  REPEATID


params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [10,20], # seems to help a little with larger problems, >3 recommended
    'n_neighbors' :[20,30] # seems to not matter much 25 and 50 look the same, 15 and 75 also
}
#maketasks(params_insta)
instancemakerparams = [{ "n_landmarks":25, "n_neighbors":50}]


params_opt = {
    'keyorder' :  ["half_step_distance",'n_iter','multiproc',"add_grammar_rules","keeptop"],
    "half_step_distance" : [True], # true clearly supperior
    "n_iter":[20], # 5 just for ez problems
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
    success = 0
    for task in range(EXPERIMENT_REPEATS):
        taskname = "%d_%d_%d_%d" % (a,b,c,task)
        fname = ".res/"+taskname
        if os.path.isfile(fname):
            completed +=1
            res = loadfile(fname)
            success += res
            if not res:
                nosucc.append(taskname)
        else: 
            nores.append(taskname)
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
    return tuple(("numgr:%d elab:%d nlab:%d rrg_iter:%d" % (d['number_of_graphs'],d['edge_labels'],d['node_labels'],d['rrg_iter'])).split(" "))

def report():
    dat= defaultdict(dict)
    nores = []
    nosucc =[]
    for a in range(len(tasklist)):
        for b in range(len(instancemakerparams)):
            for c in range(len(Optimizerparams)):
                dat[(imtostr(b),optitostr(c))][grtostr(a)] = getvalue(a,b,c, nores, nosucc)

    import pprint
    print (pandas.DataFrame(dat).to_string())
    print ("nores",nores)
    print ('nosucc',nosucc)
    #print (pandas.DataFrame(dat))df.describe().to_string()
    '''
    print ("instancemaker params:")
    pprint.pprint(instancemakerparams) 
    print ("optimizer params:")
    pprint.pprint(Optimizerparams) 
    print ("graph configurations:")
    pprint.pprint(tasklist  )

    '''





