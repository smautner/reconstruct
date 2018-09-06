import tools.imagedata as idat
import copy
#graphs = idat.loadfile("__t5__")

from collections import defaultdict 
import random

def make_library(graphs):
    one = defaultdict(set)
    two = defaultdict(set)
    zero = set()
    
    for g in graphs:
        for a,b,d in g.edges(data=True):
            al = g.node[a]['label']
            bl = g.node[b]['label']
            dl = d['label']
            
            if bl < al:
                al,bl = bl,al
                
            zero |= set([(al,dl,bl),(bl,dl,al)])
            two [dl] |= set([(al,bl)])
            two [dl] |= set([(bl,al)])
            one [al] |= set([(dl,bl)])
            one [bl] |= set([(dl,al)])
                
    return one, two, zero 




def backtrack(g,edges,one, two, zero):
    
    
    # are we done? 
    if len(edges) == 0:
        return g 
    
    # the edge we want to alter
    edges2= list(edges)
    mya, myb = edges2.pop()
    labela = g.node[mya]['label']
    labelb = g.node[myb]['label']
    
    # there are 3 cases:
    if labela == labelb == None: # zero
        
        stuff = list(zero)
        random.shuffle(stuff)
        # for all possible edge labels
        for alab,elab, blab in stuff:
            g[mya][myb]['label'] = elab
            g.node[mya]['label'] = alab
            g.node[myb]['label'] = blab
            if backtrack(g,edges2,one,two,zero):
                return g

    if labela != None and  labelb != None: # two
        
        item = (labela, labelb) if labela < labelb else (labelb, labela)
        candidates = list(two[(item)])
        random.shuffle(candidates)
        for candi in candidates: 
            g[mya][myb]['label'] = candi
            if backtrack(g,edges2,one,two,zero):
                return g
        
    else: # one
        # note: we know that exactly one neighbors label is None
        zeroguy, nonzeroguy = (mya , myb) if labelb else (myb, mya) 
        label = labela or labelb
        candidates =list(one[label])
        random.shuffle(candidates)
        for elabel,otherlabel in candidates: 
            g[mya][myb]['label']= elabel
            g.node[zeroguy]['label'] = otherlabel
            if backtrack(g,edges2,one,two,zero):
                return g
    
    # reset 
    g.node[mya]['label']=None
    g.node[myb]['label']=None
    g[mya][myb]['label']= None
    return False
    
    
def reassign_edges(graph, one, two, zero):
    # empty graph to populate
    g=graph.copy()
    for n,d in g.nodes(data=True):
        d['label']=None
    for a,b,d in g.edges(data=True):
        d['label']=None
        
    # set an oder for edges
    todo = g.edges()
    return backtrack(g,todo,one, two, zero )


def scramble(graphs): 
    print("SCRAMBLING")
    one, two, zero =  make_library(graphs)
    
    result = []
    for i in range(50): 
        result+=[reassign_edges(x,one, two, zero) for x in graphs]
    return graphs+result



###############
# previous was the population of the topology with random labels.
# now we randomly move edges around
##############

import networkx as nx
def mov_edge(graph, endpoints):
    # endpoints is {edgelabel: set(left_label,rightlabel)} # both directions are in the set
    
    # plan:
    # pick edge
    # rm, check the components,  pick random pairs of nodes to see if edg insertable
    # insert 

    edges = graph.edges(data=True)
    a,b, d = random.choice(edges)
    graph.remove_edge(a,b)
    components = list(nx.connected_components(graph))
    compa,compb = -1 , -1
    for i, li in enumerate(components): 
        if a in li:
            compa = i
        if b in li:
            compb = i
    if compa == -1 or compb == -1:
        print("FAIL")
    
    alist = components[compa]
    blist = components[compb]
    random.shuffle(alist)
    random.shuffle(blist) 
    
    #print "remove:", a,b
    for newa in alist:
        for newb in blist: 
            # new edge is not the old edge   AND   there is no edge between newa and newb
            if set([a,b]) != set([newa,newb]) and newa!= newb and  newb not in graph[newa] and d['label'] in endpoints:

                # ok if alist and blist are the same we might check edges twice. we ignore this for now
                if (graph.node[newa]['label'], graph.node[newb]['label']) in endpoints[d['label']]:
                    graph.add_edge(newa, newb, d)
                    #print "add", newa, newb
                    return graph
    
    # failed to do anything
    graph.add_edge(a, b, d)
    return graph






def edge_move(graphs, movecount=5):
    newgraphs = copy.deepcopy(graphs)
    _, lib, _ =  make_library(graphs)

    def change(gr):
        for i in range(movecount):
            gr = mov_edge(gr,lib)
    list(map(change,newgraphs))
    return newgraphs


