import pareto_funcs as paretof
import numpy as np

def greedy(graphs, target, decomposer, keepgraphs):
    """
    Return graphs with the lowest euclidean distance to the target vector.
    Also returns if one of the distances equals 0.
    """
    distances = []
    for g in graphs:
        distances.append(euclidean_distances(target, vertex_vec(g, decomposer).sum(axis=0))[0][0])
    ranked_distances = np.argsort(distances)[:keepgraphs]
    res =  [graphs[i] for i in ranked_distances]
    if distances[ranked_distances[0]] == 0:
        ## => At least 1 distance is 0 => Successful reconstruction
        return res, True
    return res, False
    

def default(graphs, costs, keepgraphs):
    """
    Take best graphs from estimators and pareto front.
    """
    costs_ranked = np.argsort(costs,axis=0)[:int(keepgraphs/6),[0,1,3]]
    want , counts = np.unique(costs_ranked,return_counts=True)
    res = [graphs[idd] for idd,count in zip( want,counts) if count > 0 ]
    dontwant = [i for i in range(len(graphs)) if i not in want]
    restgraphs = [graphs[i] for i in dontwant]
    restcosts = costs[dontwant][:,[0,1,2]]
    paretoselectedgraphs = paretof._pareto_set(restgraphs, restcosts)
    random.shuffle(paretoselectedgraphs)
    res += paretoselectedgraphs[:int(keepgraphs/2)]
    return res


def paretogreed(graphs, costs, keepgraphs):
    """
    1. choose pareto graphs 
    2. new score is the average rank over all costs
    3. choose k best of those
    """
    graphs, costs = paretof._pareto_set(graphs, costs,return_costs=True)
    costs_ranked = np.argsort(costs,axis=0).sum(axis=1)
    choosegr = np.argsort(costs_ranked)
    res = [graphs[x] for x in choosegr[:keepgraphs]]
    return res
