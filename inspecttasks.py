import reconstruct as r
import sys
filename = sys.argv[1]
graphsets = r.loadfile(filename)
import numpy as np
from structout import gprint

def size(graphs):
    return np.array([len(g) for g in graphs]).mean()

def labels(graphs):
    nlabels =[d["label"] for g in graphs for n,d in g.nodes(data=True)]
    elabels =[d["label"] for g in graphs for a,b,d in g.edges(data=True)]
    return (len(set(nlabels)),len(set(elabels)))

for graphs in graphsets:
    for i in range(5):
        gprint(graphs[i*5:i*5+5])

    # we should collect stats... somehow..
    # avg size of whole set vs last 20 
    print("avg size all:",size(graphs))
    print("avg size last 20",size(graphs[-20:]))
    print ("labels all nodelabels/edgelabels:",labels(graphs))
    print("labels last 20  n/l:",labels(graphs[-20:]))
    print ("graphs in set %d" % len(graphs))
    print ("size last 20 %s" % str([len(g) for g in graphs[-20:]]))
