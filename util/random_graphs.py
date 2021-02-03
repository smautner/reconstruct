import networkx as nx

#lolimport graphlearn01.utils.draw as draw
import eden.graph as edeng
import random

def make_random_graph(numnodes=(5,5), numedges = (5,5), maxdegree=3, allow_cycles=True):

    numnodes = random.randint(*numnodes)
    numedges = random.randint(max(numnodes-1, numedges[0]), max(numnodes-1, numedges[1]))
    active_nodes = [0]
    
    # make a connected component
    g= nx.empty_graph(numnodes)
    for end in range(1,numnodes):
        start = random.choice(active_nodes)
        while g.degree(start) >= maxdegree:
            active_nodes.remove(start)
            start = random.choice(active_nodes)
        active_nodes.append(end)
        g.add_edge(start,end)
        numedges -=1 
    if not allow_cycles:
        return g
    # first lets see what the feasible nodes are: 
    active_nodes = [x for x in active_nodes if g.degree(x) < maxdegree]

    # so, now we want to use the rest of the edges
    while numedges > 0 and len(active_nodes)>=2:
        random.shuffle(active_nodes)
        con = active_nodes[:2]
        g.add_edge(*con)
        for e in con:
            if g.degree(e)>=maxdegree: 
                active_nodes.remove(e)
        numedges -=1 
    return g
    
#g= [make_random_graph() for e in range(5)]
#so.gprint(g)


def get_zipf_label(maxx):
    n=np.random.zipf(1.15)
    while n > maxx:
        n=np.random.zipf(1.15)
    return n

def add_labels_zipf(g, nodelabels=10, edgelabels=5):
    for n,d in g.nodes(data=True):
        d['label'] = hex(get_zipf_label(nodelabels))[2:]
    for a,b,d in g.edges(data=True):
        d['label'] = hex(get_zipf_label(edgelabels))[2:]+"_"
    return g


#g= [add_labels(make_random_graph()) for e in range(5)]
#so.gprint(g)


def make_graphs_uniform(n, nodes, edges, nlabels, elabels, maxdegree):
    return [ add_labels( make_random_graph(nodes,edges,maxdegree) , nlabels,elabels )  for e in range(n)  ]

#g= make_graphs_uniform(5, (4,7), (3,9), 10,5,3 ) 
#so.gprint(g)

import numpy as np
def edgedis_allgraphs(z):
    samp = lambda x: int(np.random.normal(-1.8874912119 + 1.4263279121* x + 0.0028033789384* x**2  )+.5)
    v = samp(z)
    while v<z-1:
        v= samp(z)
    return v
def nodedis_allgraphs():
    v = int(np.random.normal(10.58124, 3.88217)+.5) 
    while v < 5:
        v = int(np.random.normal(10.58124, 3.88217)+.5) 
    return v
        
def nodedis_3deg():
    return int(np.random.triangular(5,5,13)+.5)

def edgedis_3deg(n):
    return int(np.random.triangular(n-1,n-1,n+3)+.5)




def make_graphs_real(n, nlabels, elabels, maxdegree):
    nodes = [ nodedis_3deg() for i in range(n)  ]
    edges = [ edgedis_3deg(z) for z in nodes  ]
    return [ add_labels( make_random_graph( (n,n),(e,e), maxdegree) , nlabels,elabels ) 
            for n,e in zip(nodes,edges)  ]

def add_labels_preset(g, nodelabels, edgelabels):
    random.shuffle(nodelabels)
    for nodelabel, (n,d) in zip(nodelabels, g.nodes(data=True)):
        d['label'] = str(nodelabel)#hex(nodelabel)[2:]

    random.shuffle(edgelabels)
    for edgelabel, (a,b,d) in zip(edgelabels,g.edges(data=True)):
        d['label'] = chr(edgelabel+96)#hex(edgelabel)[2:]+"_"
    return g



from scipy.stats import multinomial

real_graph_nodelabeldistribution =np.array([54659, 49901, 42444, 37924, 33794, 31740, 25053, 23488, 22326, 21313, 21181, 19053, 17362, 17333, 17173, 14112, 13745, 13475, 13304, 13091, 12412, 12069, 11973, 11901, 11405, 10978, 10798, 10266, 9903, 9339, 9328, 9233, 8809, 8354, 8321, 8208, 7987, 7911, 7743, 7603, 7379, 6980, 6964, 6597, 6386, 6375, 6331, 6215, 6178, 6154, 6151, 6090, 6051, 6035, 6033, 5994, 5935, 5834, 5618, 5617, 5511, 5491, 5391, 5353, 5290, 5043, 5036, 5022, 4906, 4898, 4853, 4824, 4742, 4683, 4614, 4593, 4591, 4575, 4558, 4541, 4509, 4452, 4433, 4429, 4406, 4379, 4343, 4339, 4316, 4259, 4218, 4093, 4089, 4022, 4012, 4003, 3994, 3987, 3929, 3910, 3890, 3759, 3674, 3664, 3643, 3623, 3501, 3412, 3396, 3335, 3322, 3265, 3240, 3206, 3204, 3154, 3104, 3061, 3060, 3029, 2981, 2943, 2784, 2776, 2757, 2700, 2688, 2644, 2636, 2635, 2623, 2604, 2600, 2593, 2590, 2590, 2582, 2507, 2467, 2429, 2388, 2199, 2198, 2173, 2143, 2056, 1792, 1765, 1743, 1704], dtype='f')
real_graph_edgelabeldistribution = np.array([141664, 67965, 49459, 41601, 26547, 23866, 15389, 13383, 10535, 7796, 5407, 5336, 4827, 4563, 3633, 2957, 2121, 1970, 1858, 1593, 1562, 1261, 1117, 1114, 1076, 1042, 914, 744, 684, 678, 643, 640, 610, 550, 501, 482, 446, 419, 379, 335, 330, 312, 266, 256, 245, 223, 138, 126, 34, 20],dtype='f')


def make_graph_strict(nodes, nlabels, elabels, maxdegree=3, dist = 'real', allow_cycles=True):
    edges =  edgedis_3deg(nodes) 

    def makestat(numlab,repeats, dist):
        ob = dist[:numlab]
        ob /= sum(ob)
        return multinomial(repeats,ob)
    
    if dist=="real":
        node_stats = makestat(nlabels, nodes, real_graph_nodelabeldistribution)
        edge_stats = makestat(elabels, edges, real_graph_edgelabeldistribution)
    
    elif dist=='uniform':
        node_stats = makestat(nlabels, nodes, np.array([1]*nlabels,dtype='f'))
        edge_stats = makestat(elabels, edges, np.array([1]*elabels,dtype='f'))

    getlab = lambda stats: [i+1 for i,e in enumerate(stats.rvs()[0]) for z in range(e) ]
            
    # DO STH WITH WITH LABELZ
    return add_labels_preset( make_random_graph( (nodes,nodes),(edges,edges), maxdegree, allow_cycles=allow_cycles) , getlab(node_stats),getlab(edge_stats) ) 

import graphlearn.lsgg_core_interface_pair as glcip
from structout import gprint


def make_graphs_static(number_of_graphs=100, size_of_graphs=5, node_labels=5, edge_labels=5, maxdeg=3,
                       labeldistribution='real', allow_cycles=True):
    l = []
    seen = {}
    failcount = 0
    while len(l)< number_of_graphs:
        g= make_graph_strict(size_of_graphs, node_labels, edge_labels, maxdeg, dist=labeldistribution, allow_cycles=allow_cycles)
        harsch = glcip.graph_hash(edeng._edge_to_vertex_transform(g.copy()),get_node_label=lambda id,node:hash(node['label']))
        if harsch not in seen:
            seen[harsch] = 1
            l.append(g)
            failcount = 0
        else:
            #gprint(g)
            failcount+=1
            assert failcount < 10, "failed 10 times in a row to generate an unseen rand graph, check your params"
                
    return l 


#g= make_graphs_real(5,10,5,3 ) 
#so.gprint(g)






