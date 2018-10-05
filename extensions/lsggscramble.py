import collections
from graphlearn3 import lsgg
import networkx as nx



import multiprocessing as mp
def funmap(f, args, poolsize=10):
    pool = mp.Pool(poolsize)
    res = pool.map(f, args)
    pool.close()
    return res

def get_labelstats(graphs):
    edges = collections.Counter()
    nodes = collections.Counter()

    for g in graphs:
        edges += collections.Counter( [  d['label'] for a,b,d in g.edges(data=True)   ] )
        nodes += collections.Counter( [  d['label'] for a,d in g.nodes(data=True)   ] )
    return edges, nodes


def enumerator(array=[0,0],cpos=0,symb=2):
    if cpos >= len(array):
        return [list(array)]
    rangestart = array[ max(cpos-1,0)  ] 
    results = []
    for i in range( rangestart, symb ):
        array[cpos] = i
        results += enumerator( array, cpos+1,symb)
    return results
            

import structout as so
def add_cip(grammar, graph):
    cip= next(grammar._cip_extraction_given_root(graph,0))
    '''
    so.gprint(cip.graph)
    print(cip.graph.graph)
    so.graph.ginfo(cip.graph)
    print (cip.__dict__)
    '''
    grammar._add_cip(cip)

def enhance(grammar,graphs, makelsgg, nodecount = 10, edgecount = 5, degree =3 ):
    edges, nodes = get_labelstats(graphs)
    edges =  [key for key, count in edges.most_common(edgecount)]
    nodes =  [key for key, count in nodes.most_common(nodecount)]
    #print "most common nodes, edges: ", nodes, edges
    #grammar = makelsgg() #lol
    enhance_HALFDIST(grammar, edges, nodes)
    #enhance_multiproc(grammar, edges, nodes, makelsgg, degree)
    return grammar

def enhance_HALFDIST(grammar,edges,nodes):

    arms = [ (a,b) for a in edges for b in nodes   ]
    add = lambda graph, arms: add_arms_add_grammar_HALFDIST(grammar, graph,arms)

    # 2 nodes: 
    graph = nx.star_graph(1)
    graph.graph['expanded'] = True
    graph._node[0]['node'] = True
    for rootlabel in nodes:
        graph._node[0]['label'] = rootlabel
        for e in edges:
            graph._node[1]['label'] = e
            graph._node[1]['edge'] = True
            add_cip(grammar,graph)

    for numnodes in range(2,4):
        graph = nx.star_graph(numnodes)
        graph.graph['expanded'] = True
        graph._node[0]['node'] = True
        combos = enumerator([0]*numnodes,symb = len(edges))
        for rootlabel in nodes:
            graph._node[0]['label'] = rootlabel
            for combo in combos:
                add(graph, [edges[e] for e in combo])


def add_arms_add_grammar_HALFDIST (grammar, graph, arms):
    for i, edge in enumerate(arms):
        graph._node[i+1]['label'] = edge
        graph._node[i+1]['edge'] = True
    add_cip(grammar,graph)




def add_arms_add_grammar (grammar, graph, arms):
    for i, (edge, node) in enumerate(arms):
        graph.node[i+1]['label'] = node
        graph[0][i+1]['label'] = edge
    add_cip(grammar,graph)


def enhance_real(grammar,edges,nodes):
    arms = [ (a,b) for a in edges for b in nodes   ]
    add = lambda graph, arms: add_arms_add_grammar(grammar, graph,arms)

    # 2 nodes: 
    graph = nx.star_graph(1)
    for rootlabel in nodes:
        graph.node[0]['label'] = rootlabel
        for arm in arms:
            add(graph, [arm])

    for numnodes in range(2,4):
        graph = nx.star_graph(numnodes)
        combos = enumerator([0]*numnodes,symb = len(arms))
        for rootlabel in nodes:
            graph.node[0]['label'] = rootlabel
            for combo in combos:
                add(graph, [arms[e] for e in combo])



def enhance_multiproc(grammar,edges,nodes, makelsgg, degree):
    arms = [ (a,b) for a in edges for b in nodes   ]
    add = lambda graph, arms: add_arms_add_grammar(grammar, graph,arms)
    # 2 nodes: 
    graph = nx.star_graph(1)
    for rootlabel in nodes:
        graph.node[0]['label'] = rootlabel
        for arm in arms:
            add(graph, [arm])
    for numnodes in range(2,degree+1):
        combos = enumerator([0]*numnodes,symb = len(arms))
        # THIS NEEDS TO BE MULTIPROCESSED
        grammarz = funmap(blubber, [ (rootlabel,nx.star_graph(numnodes),combos, arms,makelsgg)  for rootlabel in nodes], poolsize=36)
        for g2 in grammarz:
            union(grammar,g2)


def union(one_grammar, other_grammar):
    """union of grammars hehe copied from gl01 :3"""
    # all interfaces in the other
    for interface in other_grammar.productions:
        # if we dont have the others interface, we take it
        # else we have the others interface:
        if interface not in one_grammar.productions:
            one_grammar.productions[interface] = other_grammar.productions[interface]
    else:
        one_grammar.productions[interface].update(other_grammar.productions[interface])





def blubber(thing):
        root, graph, combos, arms, makelsgg = thing
        graph.node[0]['label'] = root
        grammar = makelsgg()
        for combo in combos:
            add_arms_add_grammar(grammar,graph, [arms[e] for e in combo])
        return grammar

def makelsgg():
    return  lsgg.lsgg(decomposition_args={"radius_list": [0],
                                     "thickness_list": [1]},
                 filter_args={"min_cip_count": 1,
                              "min_interface_count": 1}
                 )


