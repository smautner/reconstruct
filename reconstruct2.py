from maketasks import EXPERIMENT_REPEATS, tasklist
import os
from report import id_to_options
from util.util import jdumpfile, InstanceMaker, loadfile
from exploration import pareto
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)
logger = logging.getLogger(__name__)
import argparse

from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from graphlearn.cipcorevector import vertex_vec
from scipy.sparse import csr_matrix





def maketasks(params):
    # want a list
    combolist =[[]]
    for key in params['keyorder']:
        combolist = [  e+[value] for value in params[key]  for e in combolist ]
    return  [ {k:v for k,v in zip(params['keyorder'],configuration)} for configuration in combolist ]




######################
#  OPTIONS FOR PROBLEM GENERATOR
#####################15
# call with reconstruct.py TASKID  REPEATID
params_insta= {
    'keyorder' :  ["n_landmarks", "n_neighbors"],
    'n_landmarks' : [10], # [10,15,20], # seems to help a little with larger problems, >3 recommended
    'n_neighbors' : [100] # [50,75,100,125,200,400] # seems to not matter much 25 and 50 look the same, 15 and 75 also
    }
instancemakerparams = maketasks(params_insta)

############################
#  OPTIONS FOR SOLVER 
##############################
params_opt = {
    'keyorder' :  ["core_sizes","min_count","context_size","removeworst",'n_iter','multiproc',"add_grammar_rules","keepgraphs",
                   "squared_error", "graph_size_limiter", "cipselector_option", "cipselector_k", "use_normalization", "pareto_option"],
    "core_sizes" : None, # on exp graph ##### was [[0,2,4]]
    "removeworst":[0],
    'min_count':[2],
    "context_size":None, # you want 2 or 4 ... ##### was [2]
    "n_iter":[20], # 5 just for ez problems
    "keepgraphs":[30], # Ensure this is a multiple of 6 to not cause weird rounding errors.
    'multiproc': [4],
    "add_grammar_rules":[False],
    "squared_error": [False], # False slightly better 590:572 
    "graph_size_limiter":[ lambda x: x.max()+(int(x.std()) or 5) ], # [ lambda x: 100]
    "cipselector_option": None,
    "cipselector_k": None, # NOTE: Ensure k for option 2 is small <20. 
    "use_normalization": None, # 1 for normalization, 0 for no normalization
    "pareto_option": None
}
# Pareto Option "default": (3*5 best graphs for each category + 15 pareto front)
# Pareto Option "random":  (No pareto front and no 3*5 best graphs. Just take 30 random graphs total)
# Pareto Option "greedy":  (Instead of using the pareto front, take graphs with the lowest direct distance to the target)
# Pareto Option "pareto_only": (Instead of using the 3*5 best graphs it takes double the graphs from the pareto front.
# Pareto Option: "all": (Takes EVERY graph from the pareto front)
parser = argparse.ArgumentParser()
parser.add_argument('--core_sizes', nargs='*', type=int, default=[0,1,2], 
                    help='Core sizes/Radii')
parser.add_argument('--context_size', nargs=1, type=float, default=[1],
                    help='Context sizes/Thickness')
parser.add_argument('--cipselector_option', nargs=1, type=int, default=[1], ## Change this back
                    choices=[0, 1, 2],
                    help='1: Take k best from all, 2: Take k best from each current cip')
parser.add_argument('--cipselector_k', nargs=1, type=int, default=[100],
                    help='k for Cipselector')
parser.add_argument('--pareto_option', nargs=1, type=str, default=['greedy'],
                    choices=['default', 'random', 'greedy', 'pareto_only', 'all'],
                    help='Pareto option for optimization')
parser.add_argument('--use_normalization', nargs=1, type=int, default=[1], choices=[1,0],
                    help='If 1, normalization will be applied for cipselection')
parser.add_argument('--min_count', nargs=1, type=int, default=[2], 
                    help='Also called min_cip')
parser.add_argument('--graph_size_limiter', nargs=1, type=int, default=[1], choices=[1,0],
                    help='If 0, graph size limiter is only used with a graphs >100')
parser.add_argument('--taskid', nargs=1, type=int, default=[0])
parser.add_argument('--resprefix', nargs=1, type=str, default=['.res'],
                    help='Output folder')
parser.add_argument('-c', '--chem', action='store_true', 
                    help='If used, chemtasks will be executed, not required for reportchem.')
parser.add_argument('--max_decompose_radius', nargs=1, type=int, default=[1],
                    help='Max radius for decompose neighborhood')
parsed_args = vars(parser.parse_known_args()[0])
taskid = parsed_args.pop('taskid')[0]
use_chem = parsed_args.pop('chem')
resprefix = parsed_args.pop('resprefix')[0]
use_graph_size_limiter = parsed_args.pop('graph_size_limiter')[0]
if not use_graph_size_limiter:
    params_opt['graph_size_limiter'] = [lambda x: 100]
parsed_args['core_sizes'] = [parsed_args['core_sizes']]
max_decompose_radius = parsed_args.pop('max_decompose_radius')[0]
params_opt.update(parsed_args)

Optimizerparams = maketasks(params_opt)




##################################33
# RUNNING 
############################

def decompose(x):
    return decompose_neighborhood(x, max_radius=max_decompose_radius)

def reconstruct_and_evaluate(target_graph,
                                landmark_graphs,
                                desired_distances,
                                ranked_graphs,
                                **args):
    decomposer = decompose
    # genmaxsize= np.average([g.number_of_nodes() for g in landmark_graphs]) * 1.3 ## Currently not used.
    optimizer = pareto.LocalLandmarksDistanceOptimizer(decomposer=decomposer, **args)
    target_graph_vector = csr_matrix(vertex_vec(target_graph, decomposer).sum(axis=0))
    # providing target, prints real distances for all "passing" creations
    res = optimizer.optimize(landmark_graphs, desired_distances, ranked_graphs,
                             target_graph_vector=target_graph_vector) #,target=target_graph)
    return res


if __name__=="__main__":


    #exit()
    #print(sys.argv[-1])
    #args = list(map(int, sys.argv[-1].strip().split(" ")))

    # ok need to run this on the cluster where i only have a task id...
    # the queickest way to hack this while still being compatible with the old crap
    # is using the maketasks function defined above...
    taskfilename = '.tasks'
    if use_chem:
        print("TEST")
        taskfilename = '.chemtasks'
        tasklist =list(range(13)) # chem stuff

    arg = int(taskid)-1 # was [-1]
    args= id_to_options(tasklist=tasklist)[arg]

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
    elif not os.path.exists(resprefix):
        os.makedirs(resprefix)

    im =  InstanceMaker(**im_params).fit(graphs, EXPERIMENT_REPEATS)
    res = im.get(run_id)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = res
    result = reconstruct_and_evaluate( target_graph,
            landmark_graphs,
            desired_distances,
            ranked_graphs,
            **optimizerargs)

    jdumpfile(result, filename)   




