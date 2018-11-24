from util import random_graphs as rg
import random
from collections import defaultdict
import pandas
import os
import sys
import numpy as np
from util.util import jdumpfile, jloadfile, InstanceMaker, dumpfile, loadfile
from util import rule_rand_graphs as rrg
from exploration import pareto
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)
logger = logging.getLogger(__name__)


'''
USAGE:
    python3 reconstruct.py  to generate problem instances
    fish runall.sh  to run with parallel
    python3 -c "import reconstruct as r; r.report()"   to see result
'''


def maketasks(params):
    # want a list
    combolist =[[]]
    for key in params['keyorder']:
        combolist = [  e+[value] for value in params[key]  for e in combolist ]
    return  [ {k:v for k,v in zip(params['keyorder'],configuration)} for configuration in combolist ]



#############################################
##  OPTIONS FOR GRAPHS
##########################################

EXPERIMENT_REPEATS = 50
# 1. param dict

params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels","allow_cycles","labeldistribution","maxdeg","rrg_iter"],
    'allow_cycles':[False], # cycles are very bad
    'number_of_graphs': [30],
    'size_of_graphs' :[8] ,
    'node_labels' : [4],
    'edge_labels' : [2], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] ,# real is unnecessary
    'maxdeg':[3],
    # rule rand graphs , iter argument ,  
    #0 means just use the rand graphs, a little hacky but works for now
    'rrg_iter':[3]
}

#explore nodelabels, rrg , degree number of start graphs
import copy
if True:
    _nl = copy.deepcopy(params_graphs)
    _nl["node_labels"]=[2,4,6,8,10,12,14]
    _rrg = copy.deepcopy(params_graphs)
    _rrg["rrg"]=[2,3,4,5]
    _degre = copy.deepcopy(params_graphs)
    _degre["maxdeg"]=[3,4,5]
    _sgr = copy.deepcopy(params_graphs)
    _sgr['number_of_graphs']=[20,30,40,50]
    tasklist = [ b  for thing in [_nl,_rrg,_degre,_sgr] for b in maketasks(thing)  ]
    tasknamezz= [ ("nodelabels",x) for x in [2,4,6,8,10,12,14]] + [('rrg',x) for x in [2,3,4,5]]+[('maxdeg',x) for x in [3,4,5]]+[('numgr',x) for x in [20,30,40,50]]

else:
    tasklist  = maketasks(params_graphs ) # boring task list


######################
#  OPTIONS FOR PROBLEM GENERATOR
#####################15
# call with reconstruct.py TASKID  REPEATID
params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [10], # seems to help a little with larger problems, >3 recommended
    'n_neighbors' :[100] # seems to not matter much 25 and 50 look the same, 15 and 75 also
    }
instancemakerparams =maketasks(params_insta)

############################
#  OPTIONS FOR SOLVER 
##############################
params_opt = {
    'keyorder' :  ["core_sizes","min_count","context_size","removeworst",'n_iter','multiproc',"add_grammar_rules","keeptop","squared_error","graph_size_limiter"],
    "core_sizes" : [[0,1,2,3,4]], # on exp graph
    "removeworst":[0],
    'min_count':[1],
    "context_size":[2], # you want 2 or 4 ...
    "n_iter":[20], # 5 just for ez problems
    "keeptop":[5], # 5+  15 pareto things
    'multiproc': [8],
    "add_grammar_rules":[False],
    "squared_error": [False], # False slightly better 590:572 
    "graph_size_limiter":[ lambda x: x.max()+(int(x.std()) or 5) ]
}

if False:
    #%core sizes vs insterface size might tell a story, artificial: thick2 core 0 ,,
    #%edge_as_if core 012(old coordinates);; r0,1 , thickness 1 

    _a = copy.deepcopy(params_opt)
    _a["context_size"] = [4]
    _a["core_sizes"] = [[0]]
    _b = copy.deepcopy(params_opt)
    _b["context_size"] = [1]
    _b["core_sizes"] = [[0,2,4]]
    _c = copy.deepcopy(params_opt)
    _c["context_size"] = [2]
    _c["core_sizes"] = [[0,2]]
     #%grammar gen arguments vs recovery arguments table fix thickness... smaller
    #%core, larger, same
    _d = copy.deepcopy(params_opt)
    _d["context_size"] = [2]
    _d["core_sizes"] = [[0,2],[0],[0,2,4]]

    Optimizerparams = [b for thing in [_a,_b,_c,_d] for b in maketasks(thing)]
else:
    Optimizerparams = maketasks(params_opt)



###################################
# WRITING TASK FILES 
####################################
# 3. loop over task


def maketsk(args):
    rrg_iter = args.pop("rrg_iter")
    graphs = rg.make_graphs_static(**args)
    if rrg_iter > 0:
        graphs = rrg.rule_rand_graphs(graphs, numgr=550,iter=rrg_iter)[0]
    return graphs
def make_task_file():
    import extensions.lsggscramble  as scram
    data = scram.funmap(maketsk, tasklist,poolsize=20)
    dumpfile(data, ".tasks")
    #dumpfile([ rg.make_graphs_static(maxdeg=3, **args) for args in tasklist], ".tasks")

def load_chem(AID):
    import json
    import networkx.readwrite.json_graph as sg
    import networkx as nx
    import exploration.pareto as pp
    from structout import gprint
    with open(AID, 'r') as handle:
        js = json.load(handle)
        res = [sg.node_link_graph(jsg) for jsg in js]
        res = [g for g in res if len(g)> 2] 
        res = [g for g in res if nx.is_connected(g)]  # rm not connected crap
        for g in res:g.graph={}
        zz=pp.MYOPTIMIZER()
        res2 = list(zz._duplicate_rm(res))
        print ("duplicates in chem files:%d"% (len(res)-len(res2)))
        print (zz.collisionlist)
        #for a,b in zz.collisionlist:
        #    gprint([res[a],res[b]])
        zomg = [(len(g),g) for g in res]
        zomg.sort(key=lambda x:x[0])
        cut = int(len(res)*.1)
        res2 = [b for l,b in zomg[cut:-cut]]
    return res2

def get_chem_filenames():
    # these are size ~500
    files="""AID1224837.sdf.json  AID1454.sdf.json  AID1987.sdf.json  AID618.sdf.json     AID731.sdf.json     AID743218.sdf.json  AID904.sdf.json AID1224840.sdf.json  AID1554.sdf.json  AID2073.sdf.json  AID720709.sdf.json  AID743202.sdf.json  AID828.sdf.json"""
    # these are size ~4000
    files='''AID119.sdf.json
            AID1345082.sdf.json
            AID588590.sdf.json
            AID624202.sdf.json
            AID977611.sdf.json'''
    files = files.split()
    return files

def make_chem_task_file():
    files = get_chem_filenames()
    res=[]
    for f in files: 
        stuff =load_chem("chemsets/"+f)
        random.shuffle(stuff)
        res.append(stuff)
    dumpfile(res, ".chemtasks")




def id_to_options(tasklist=tasklist):
    params_args = {"keyorder":[3,2,1,0], # 3 first -> order works out ineval
                        0:range(len(tasklist)),
                        1:range(len(instancemakerparams)),
                        2:range(len(Optimizerparams)),
                        3:range(EXPERIMENT_REPEATS),
                        }
    args = maketasks(params_args)
    return args


#####################
# EVAL OUTPUT FORMAT
#######################
def defaultformatter(paramsdict, instance_dict):
    res =[]
    for k in paramsdict['keyorder']:
        if len(paramsdict[k] )> 1:
            #  interesting key
            res.append("%s:%s " % ( k[:4],str(instance_dict[k])) )
    res=  tuple(res) or "lol"
    return res

def imtostr(im):
    d=instancemakerparams[im]
    return "marks:%d neigh:%d" % (d["n_landmarks"], d["n_neighbors"])

def optitostr(op):
    d=Optimizerparams[op]
    return defaultformatter(params_opt,d)

def grtostr(gr):
    d = tasklist[gr]
    return defaultformatter(params_graphs,d)
    #return tuple(("Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])).split(" "))
    #return tuple(("elab:%d nlab:%d" % (d['edge_labels'],d['node_labels'])).split(" "))
    #return tuple(("elab:%d nlab:%d graphs:%d rrg_it:%d" % (d['edge_labels'],d['node_labels'],d['number_of_graphs'],d['rrg_iter'])).split(" "))



##############################
# EVALUATING
##########################

def getvalue(p, nores, nosucc, folder): # nosucc and nores are just collecting stats
    completed = 0
    allsteps=[-1]
    success = 0
    for task in range(EXPERIMENT_REPEATS):
        taskname = "%d" % (p+task)
        fname = folder+"/"+taskname
        if os.path.isfile(fname):
            completed +=1
            res, steps = jloadfile(fname)
            success += res
            if not res:   # FAIL
                nosucc.append(taskname)
            else:       # success -> remember step count
                allsteps.append(steps)
        else: 
            nores.append(taskname)
    allsteps = np.array(allsteps)
    return success,  allsteps.max()

def report(folder = '.res', tasklist=None):

    problems = id_to_options(tasklist= tasklist)

    dat= defaultdict(dict)
    nores = []
    nosucc =[]
    for p in range(0,len(problems),EXPERIMENT_REPEATS):
        a,b,c,_ = [ problems[p][k] for k in [0,1,2,3]]
        im = imtostr(b)
        gr = grtostr(a)
        op = optitostr(c)
        x,y = tasknamezz[a]
        dat[x][y]= getvalue(p, nores, nosucc, folder)

    #mod = lambda x : str(x).replace("_",' ')
    print ("nores",nores)
    print ('nosucc',nosucc)
    print ("sumsuccess:", sum([int(a) for c in dat.values() for a,b in c.values()]))
    print ("maxrnd:", max([int(b) for c in dat.values() for a,b in c.values()]))

    print (pandas.DataFrame(dat).to_string()) 
    print (pandas.DataFrame(dat).to_latex()) 



##################################33
# RUNNING 
############################

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
    if sys.argv[1]=="maketasks":
        print("writing task file...")
        make_task_file()
        exit()
    elif sys.argv[1]=="report":
        report('.res',tasklist)
        exit()
    elif sys.argv[1]=="reportchem":
        tasklist = get_chem_filenames()
        report('.chemres',tasklist)
        exit()
    elif sys.argv[1]=="maketaskschem":
        print("writing task file...")
        make_chem_task_file()
        exit()
    else:
        #print(sys.argv[-1])
        #args = list(map(int, sys.argv[-1].strip().split(" ")))
        
        # ok need to run this on the cluster where i only have a task id...
        # the queickest way to hack this while still being compatible with the old crap 
        # is using the maketasts function defined above...
        taskfilename = '.tasks'
        resprefix='.res'
        if sys.argv[-2] == 'chem':
            taskfilename = '.chemtasks'
            resprefix='.chemres'
            tasklist=list(range(13)) # chem stuff

        arg = int(sys.argv[-1])-1
        args=id_to_options(tasklist=tasklist)[arg]

    #OPTIONS FOR GRAPHS
    task = loadfile(taskfilename)
    task_id = args[0] 
    graphs = task [task_id]

    # landmark graphs , n neighs
    im_param_id= args[1]
    im_params = instancemakerparams[im_param_id]

    # OPTIONS FOR OPTIMIZER
    optimizer_para_id = args[2]
    optimizerargs = Optimizerparams[optimizer_para_id]

    logger.debug(im_params)
    logger.debug(tasklist[task_id])
    logger.debug(optimizerargs)

    run_id =args[3] 

    filename = "%s/%d" % (resprefix,arg)
    if os.path.isfile(filename):
        print ("FILE EXISTS")
        exit()

    im =  InstanceMaker(**im_params).fit(graphs, EXPERIMENT_REPEATS)
    res = im.get(run_id)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = res
    result = reconstruct_and_evaluate( target_graph,
            landmark_graphs,
            desired_distances,
            ranked_graphs,
            **optimizerargs)

    jdumpfile(result, filename)   




