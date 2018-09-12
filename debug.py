from util import random_graphs as rg
from sklearn.neighbors import NearestNeighbors

from eden.util import configure_logging
import logging
from util.util import InstanceMaker








####################
# run and get best result
###################
from eden.graph import Vectorizer
from exploration import pareto


def get_best_distance(pareto_set_graphs, target_graph):
    vec = Vectorizer(r=3, d=6,normalization=False, inner_normalization=False)
    pareto_set_vecs = vec.transform(pareto_set_graphs)
    #nn = NearestNeighbors(n_neighbors=len(pareto_set_graphs)).fit(pareto_set_vecs) 
    nn = NearestNeighbors(n_neighbors=1).fit(pareto_set_vecs)  # 1 is enough!
    reference_vec = vec.transform([target_graph])
    distances, neighbors = nn.kneighbors(reference_vec,return_distance=True)
    distances = distances[0]
    neighbors = neighbors[0]
    return min(distances)






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

import structout as so
def test_randgraphs():

    #make_graphs_static(n,ncnt,nlab,elab,maxdeg=3, labeldistribution='real'):
    graphs = rg.make_graphs_static(10, # how many to generate
                                    5, # graph size
                                    5, # node-labelcount
                                    2, # edgelabelcount
                                    labeldistribution='uniform')
    so.graph.ginfo(graphs[0])
    while graphs:
        so.gprint(graphs[:3],edgelabel='label')
        graphs = graphs[3:]
    return graphs


def test_instancemaker():
    graphs = rg.make_graphs_static(7, # how many to generate
                                   5, # graph size
                                   5, # node-labelcount
                                   2, # edgelabelcount
                                   labeldistribution='uniform')
    im =  InstanceMaker(n_landmarks=3, n_neighbors=6).fit(graphs,ntargets=2)
    landgraphs,des_dist,rest, target = im.get()
    print("landmarks")
    so.gprint(landgraphs, edgelabel='label')
    print("des dist")
    print(des_dist)
    print("target")
    so.gprint(target)
    print("rest")
    so.gprint(rest,edgelabel='label')

import random
def test_grammar():

    graphs = rg.make_graphs_static(7, # how many to generate
                                   5, # graph size
                                   5, # node-labelcount
                                   2, # edgelabelcount
                                   labeldistribution='uniform')
    optimizer = pareto.LocalLandmarksDistanceOptimizer()
    optimizer.enhance_grammar(graphs)
    print(optimizer.grammar)
    keys = list(optimizer.grammar.productions.keys())
    random.shuffle(keys)
    print ("start w grammar")
    for k in  keys[:10]:
        cips = list(optimizer.grammar.productions[k].values())
        so.gprint([c.graph for c in cips], color=[[c.core_nodes,c.interface_nodes] for c in cips])
        #so.graph.ginfo(cips[0].graph)
        print(cips[0].__dict__)



def test_neighexpansion():
    graphs = rg.make_graphs_static(7, # how many to generate
                                   4, # graph size
                                   3, # node-labelcount
                                   2, # edgelabelcount
                                   labeldistribution='uniform')
    optimizer = pareto.LocalLandmarksDistanceOptimizer()
    optimizer.enhance_grammar(graphs)
    neighs = list(optimizer.grammar.neighbors(graphs[0]))

    so.gprint(graphs[0])
    so.gprint(neighs)


#test_instancemaker()
#test_randgraphs()
#test_grammar()
#test_neighexpansion()



def test_pareto():
    configure_logging(logging.getLogger(),verbosity=2)
    graphs = rg.make_graphs_static(100, # how many to generate
                                   5, # graph size
                                   4, # node-labelcount
                                   2, # edgelabelcount
                                   labeldistribution='uniform',
                                   allow_cycles=False)

    im =  InstanceMaker(n_landmarks=5, n_neighbors=50).fit(graphs,ntargets=2)

    optimizer = pareto.LocalLandmarksDistanceOptimizer(n_iter=7, context_size=1,multiproc=True)
    landmark_graphs, desired_distances, ranked_graphs, target_graph = im.get()
    NONE = optimizer.optimize(landmark_graphs,
                              desired_distances,
                              ranked_graphs,
                              #start_graph_list=[landmark_graphs[0]])
                              start_graph_list=landmark_graphs)
    #NONE = optimizer.optimize(landmark_graphs, desired_distances, ranked_graphs, start_graph_list=landmark_graphs)
    #print("resulting set:")
    #so.gprint(reconstructions, edgelabel='label')
    return  None
    #return get_best_distance(reconstructions, target_graph)

test_pareto()
