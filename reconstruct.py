from util import random_graphs as rg
import os
import sys

from util.util import jdumpfile, jloadfile, InstanceMaker
####################
# run and get best result
###################
from exploration import pareto




def reconstruct_and_evaluate(target_graph,
                                landmark_graphs,
                                desired_distances,
                                ranked_graphs,
                                args):
    optimizer = pareto.LocalLandmarksDistanceOptimizer(**args)
    res = optimizer.optimize(landmark_graphs,desired_distances,ranked_graphs)


    return res
   




####################
# run
#################



max_size_frontier=None
args = dict(
    r=3,d=6,
    min_count=1, # grammar mincount
    context_size=1, # grammar context size
    expand_max_n_neighbors=None,
    n_iter=500,  
    expand_max_frontier= -1, # max_size_frontier*5, # -1 or 0 might work.. 
    max_size_frontier=max_size_frontier, 
    adapt_grammar_n_iter=None)

n_neighbors =  20

graphsize = 5
graphcount = 60 # ?? wat
nlabel = 10
elabel = 5
maxdeg = 3
n_exps = 1

labeldistribution= 'uniform' # uniform or real
problems = [ (gct,gsi,nlabel,elabel) 
                                     for gct in [100,500]  # graphs
                                     for gsi in [5]             # graphsize
                                     for nlabel in [3,10,15]    # number of node labels?
                                     for elabel in [3]          # number of edgelabels
                                     ]


if __name__=="__main__":
    
    resu=[]
    graphcount, graphsize,nlabel,elabel = problems[int(sys.argv[1])]
    FNAME= "%d_%d_%d_%d" % (graphcount,graphsize,nlabel,elabel)

    graphs = rg.make_graphs_static(graphcount+n_exps,graphsize,nlabel,elabel,labeldistribution=labeldistribution)
    im =  InstanceMaker(n_landmarks=max_size_frontier, n_neighbors=n_neighbors).fit(graphs, n_exps)
    for i in range(n_exps):
        res = im.get()
        landmark_graphs, desired_distances, ranked_graphs, target_graph = res
        resu.append(reconstruct_and_evaluate( target_graph, landmark_graphs, desired_distances, ranked_graphs, args))


    jdumpfile(resu, "EXPL/" + FNAME)   #!!!



#######################################
# Report 
#########################


from collections import defaultdict
import pandas
def getvalue(a,b,c,d):
    fname = "EXPL/%d_%d_%d_%d" % (a,b,c,d)
    if os.path.isfile(fname):
        res = jloadfile(fname)
        #return np.mean(np.array(loadfile("EXPL/%d_%d_%d_%d" % (a,b,c,d))))

        return sum([1 for e in res if e < 0.01])
    else:
        return None
    #return np.mean(np.array(loadfile("EXPL/%d_%d_%d_%d" % (a,b,c,d))))


def report():
    dat= defaultdict(dict)
    for a,b,c,d in problems:
        dat[(a,b)][(c,d)] = getvalue(a,b,c,d)
    print("labeldistr", labeldistribution)
    return pandas.DataFrame(dat)




