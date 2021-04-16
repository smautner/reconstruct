'''

import os
from util.util import jdumpfile, InstanceMaker, loadfile
from exploration import pareto
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)
logger = logging.getLogger(__name__)
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from graphlearn.cipcorevector import vertex_vec
from scipy.sparse import csr_matrix

'''


doc='''
--n_landmarks int default:10
--n_neighbors int default:100

--out str default:res/out.txt
--in str default:.taskfile
--taskid int default:0

--maxcoresize int default:2 
--context_size int default:1
--min_count int default:2             not sure what this is 

--cipselector int default:2           0 -> k is on populationlevel , 1 -> k is io graphlevel ,2 -> k is on ciplevel
--cipselector_k int default:1
--pareto str default:default          ['default', 'random', 'greedy', 'pareto_only', 'all']
--skip_normalization
--graph_size_limiter int default:1    # 0 -> no limit , 1 -> mean+sigma ,  n -> n  
--lookahead_radius int default:1  # radius for lookahead system
'''


import dirtyopts as opts
args = opts.parse(doc)
print(args.lookahead_vectorizer)


##################################33
# RUNNING 
############################

def decompose(x):
    return decompose_neighborhood(x, max_radius=args.lookahead_radius)

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


'''

taskid = parsed_args.pop('taskid')[0]
use_chem = parsed_args.pop('chem')
resprefix = parsed_args.pop('resprefix')[0]
use_graph_size_limiter = parsed_args.pop('graph_size_limiter')[0]
if not use_graph_size_limiter:
    params_opt['graph_size_limiter'] = [lambda x: 100]
parsed_args['core_sizes'] = [parsed_args['core_sizes']]
max_decompose_radius = parsed_args.pop('max_decompose_radius')[0]

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


{
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
'''
