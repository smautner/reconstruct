from util import random_graphs as rg
from eden.util import configure_logging
import logging
from util.util import InstanceMaker
from exploration import pareto
import structout as so

from eden.util import configure_logging  
import logging
configure_logging(logging.getLogger(),verbosity=2)

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
    return  None


def test_estimators():
    from util.util import loadfile
    import reconstruct
    import exploration.cost_estimator as costor
    graphs = loadfile(".tasks")[0]
    im_param_id= 0
    im_params = reconstruct.instancemakerparams[im_param_id]

    im =  InstanceMaker(**im_params).fit(graphs, 10)
    esti = costor.DistRankSizeCostEstimator(multiproc=True)
    a,b,c, target =im.get(0)
    esti.fit(b,a,c)

    ex = loadfile("gr")
    print (esti.decision_function([target]))
    print (esti.decision_function([ex]))
    so.gprint([ex,target])


test_estimators()


#test_pareto()
#test_instancemaker()
#test_randgraphs()
#test_grammar()
#test_neighexpansion()

