import logging
import time
import random
import queue
import heapq
import numpy as np
from eden.util import timeit
from graphlearn.cipcorevector import LsggCoreVec as lsgg
from toolz.curried import compose, map, concat
from exploration.pareto_funcs import _manage_int_or_float
logger = logging.getLogger(__name__)
import structout as so
from exploration import pareto_options, pareto_funcs as paretof, cost_estimator as costs
from extensions import lsggscramble as lsggs
from sklearn.metrics.pairwise import euclidean_distances

import eden.graph
import itertools
import multiprocessing

import graphlearn.lsgg_core_interface_pair as glcip

from sklearn.preprocessing import normalize
from collections import defaultdict

from graphlearn.cipcorevector import vertex_vec
from scipy.sparse import csr_matrix


def score_productions(target_csr, current_csr, productions, use_normalization):
    if use_normalization:
        target_csr = normalize(target_csr, axis=1)
        predicted_vectors = [csr_matrix(normalize(current_csr - curent_cip.core_vec + con_cip.core_vec, axis=1)).T
                            for curent_cip, con_cip in productions] # if con_cip <= self.genmaxsize
    else:
        predicted_vectors = [csr_matrix(current_csr - curent_cip.core_vec + con_cip.core_vec).T
                            for curent_cip, con_cip in productions]
    scores = np.dot(target_csr, predicted_vectors)
    return scores


def new_cipselector0(current_cips_congrus, target_graph_vector, current_graph_vector, use_normalization, cipselector_k):
    """
    Option 0 for the new cipselector. Using this option we will be able to generate a total of k productions.

    Args:
      current_cips_congrus (list): [(current_cip, concip), (), ...]
      target_graph_vector (matrix): Vector of the target graph
      current_graph_vector (matrix): Vector of the current graph 
    """
    target_csr = target_graph_vector
    scores = score_productions(target_csr, current_graph_vector, current_cips_congrus, use_normalization)
    return [(sco, prd) for sco, prd in zip(scores, current_cips_congrus)]


def new_cipselector(current_cips_congrus, target_graph_vector, current_graph_vector, use_normalization, cipselector_k):
    """
    Option 1 for the new cipselector. For each cippair it calculates
    target * (current_graph - current_cip + con_cip) and returns the k best of them.

    Args:
      current_cips_congrus (list): [(current_cip, concip), (), ...]
      target_graph_vector (matrix): Vector of the target graph
      current_graph_vector (matrix): Vector of the current graph 
    """
    target_csr = target_graph_vector
    scores = score_productions(target_csr, current_graph_vector, current_cips_congrus, use_normalization)
    result = np.argsort(scores)[-cipselector_k:]
    return [current_cips_congrus[x] for x in result]
    

def new_cipselector2(current_cips_congrus, target_graph_vector, current_graph_vector, use_normalization, cipselector_k):
    """
    Option 2 for the new cipselector. Finds the best concip
    for each current_cip and returns the k best of them.

    Args:
      current_cips_congrus (list): [(current_cip, concip), (), ...]
      target_graph_vector (matrix): Vector of the target graph
      current_graph_vector (matrix): Vector of the current graph
    """
    target_csr = target_graph_vector
    result = []
    scores = score_productions(target_csr, current_graph_vector, current_cips_congrus, use_normalization)
    d = defaultdict(list) # Elements are (similarity, cippair)
    for s,p in zip(scores, current_cips_congrus):
        d[p[0].interface_hash].append((s, p))
    for value in d.values():
        tmp = sorted(value, reverse=True, key=lambda x: x[0])[:cipselector_k]
        result += [i[1] for i in tmp]
    return result

def calc_average(l):
    """
    Small function to mitigate possibility
    of an empty list of average productions.
    """
    if len(l) == 0:
        return 0
    return sum(l)/len(l)
    

class hashvec(object):

    def __init__(self, vectorizer, multiproc = 1):
        self.vectorizer = vectorizer
        self.multiproc = multiproc


    def grouper(self, n, iterable):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def vectorize_chunk(self,chunk):
        feature_dicts = [ self.vectorizer._transform(graph) for graph in chunk]
        def hashor(fd):
            k= sorted(fd.keys())
            v = [fd[kk] for kk in k]
            return hash(tuple(k+v))
        hashed_features = [ hashor(fd) for fd in feature_dicts]
        return hashed_features
    
    def vectorize_chunk_glhash(self,chunk):
        return [glcip.graph_hash(eden.graph._edge_to_vertex_transform(g),2**20-1,node_name_label=lambda id,node:hash(node['label'])) for g in chunk]

    def vectorize_multiproc(self, graphs):
        with multiprocessing.Pool(self.multiproc) as p:
            res = (p.map(self.vectorize_chunk, self.grouper(1000,graphs)))
        return itertools.chain.from_iterable(res)

    def vectorize(self,graphs):
        if self.multiproc>1:
            return self.vectorize_multiproc(graphs)
        else:
            return self.vectorize_chunk(graphs)



class cheater(object):
    def __init__(self,target):
            self.vectorizer = costs.Vectorizer()
            self.cheatreference = self.vectorizer.transform([target])
    def cheat_calc(self,others):
            return costs.euclidean_distances(self.cheatreference, self.vectorizer.transform(others))[0]



class MYOPTIMIZER(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            grammar=None,
            multiobj_est=None,
            n_iter=19,
            keepgraphs= 30,
            random_state=1,multiproc=True, target_graph_vector=None, target=None, removeworst=0, pareto_option=1, use_normalization=1, decomposer=None, cipselector_k=0):
        """init."""
        self.grammar = grammar
        self.keepgraphs = keepgraphs
        self.multiobj_est = multiobj_est
        self.n_iter = n_iter
        random.seed(random_state)
        self.multiproc = multiproc
        self.hash_vectorizer = hashvec(eden.graph.Vectorizer(normalization=False,r=2,d=2),multiproc=multiproc)
        self.cheat = False
        self.seen_graphs = {}
        self.queues  = [ list() for i in range(4)]
        self.prefilter_kick= removeworst
        self.target_graph_vector = target_graph_vector
        self.cipselector_k = cipselector_k
        self.use_normalization= use_normalization
        self.pareto_option = pareto_option
        self.decomposer = decomposer
        if target:
            self.cheat= True
            self.cheater = cheater(target)



    def checkstatus(self,costs,graphs):
        g = [graphs[e] for e in np.argmin(costs,axis=0)]
        #logger.debug([len(e)for e in g])
        logger.debug(so.graph.make_picture(g,edgelabel='label',size=10))
        logger.log(10, [x.number_of_nodes() for x in g])
        if min(costs[:,0]) == 0:
            #logger.debug(graphs[np.argmin(costs[:,0])].graph['history'])
            return True
        return False

    @timeit
    def optimize(self, graphs):
        """Optimize iteratively."""
        starttime= time.time()
        averages = []
        assert (self.grammar.is_fit())
        assert (self.multiobj_est.is_fit())

        # init
        self.seen_graphs = {}
        for g in graphs:
            g.graph['history']=[]

        # main cycle
        for i in range(self.n_iter):
            logger.debug("++++++++  START OPTIMIZATION STEP %d +++++++" % i)
            graphs, status, avg_productions= self.optimize_step(graphs)
            if not avg_productions == None: # In case avg == 0 it should still be added
                averages.append(avg_productions)
            if status:
                return True,i,time.time()-starttime, calc_average(averages)
            if not graphs:
                logger.debug("ran out of graphs")
                return False,i, time.time()-starttime, calc_average(averages)

        costs = self.get_costs(graphs)
        return self.checkstatus(costs, graphs),i, time.time()-starttime, calc_average(averages)

    def optimize_step(self, graphs):
        # filter, expand, chk duplicates
        step_start_time = time.time()
        graphlen_start = len(graphs)
        graphs, status = self.filter_by_cost(graphs)
        if status: return [],True,None
        graphlen_filter = len(graphs)
        logger.log(10, f"cost_filter: Got {graphlen_start} graphs, reduced to {graphlen_filter} ({time.time()-step_start_time})")
        num_graphs = len(graphs)
        if self.grammar.cipselector == new_cipselector0:  ### SPECIAL CASE
            graphs = self._expand_neighbors2(graphs)
        else:
            graphs = self._expand_neighbors(graphs)
        graphlen_expand = len(graphs)
        avg_productions = graphlen_expand/graphlen_filter
        logger.log(10, f"graph generation: Got {avg_productions} per graph. ({time.time()-step_start_time})")
        graphs = self.duplicate_rm(graphs)
        logger.log(10, f"duplicate_rm: {graphlen_expand} -> {len(graphs)} graphs. ({time.time()-step_start_time})")
        return graphs, status, avg_productions

   

    def filter_by_cost(self,graphs):
        """expand "keepgraphs" graphs, divided between top graphs in everything
        and pareto front, discard rest"""
        keepgraphs = self.keepgraphs

        costs = self.get_costs(graphs)
        status = self.checkstatus(costs, graphs)
        if status:
            # Some graph has distance == 0
            return graphs, True
                    
        if len(graphs) <= self.keepgraphs:
            # Only few graphs remaining so just return all of them.
            logger.log(10, "cost_filter: keep all graphs")
            return graphs, False
        
##        elif self.prefilter_kick!=0:
##            # DELETE THE 25% worst in each category
##            assert False
##            costs_ranked = np.argsort(costs,axis=0)[-int(len(graphs)*self.prefilter_kick):]
##            trash = np.unique(costs_ranked)
##            keep =  [i for i in range(len(graphs)) if i not in trash]
##            graphs = [graphs[i] for i in keep]
##            costs = costs[keep]

        elif self.pareto_option == 'greedy':
            # Return graphs with the lowest euclidean distance to the target vector
            return pareto_options.greedy(graphs, self.target_graph_vector, self.decomposer, keepgraphs)

        elif self.pareto_option == "random":
            # Return randomly selected graphs without any application of pareto.
            return random.sample(graphs, keepgraphs), False

        elif self.pareto_option == "default":
            # Take best graphs from estimators and pareto front
            return pareto_options.default(graphs, costs, keepgraphs), False
        
        elif self.pareto_option == "paretogreed":
            # 1. choose pareto graphs 
            # 2. new score is the average rank over all costs
            # 3. choose k best of those 
           return pareto_options.paretogreed(graphs, costs, keepgraphs), False
        
        paretoselectedgraphs = paretof._pareto_set(graphs, costs)
        random.shuffle(paretoselectedgraphs)

        if self.pareto_option == "pareto_only":
            # Return only graphs from the pareto front
            return paretoselectedgraphs[:keepgraphs], False
        
        elif self.pareto_option == "all":
            # Return ALL graphs from the pareto front
            return paretoselectedgraphs, False
        else:
            raise ValueError("Invalid Pareto Option")

##        # DEBUG TO SHOW THE REAL DISTANCE
##        if self.cheat:
##            print ("real distances for all kept graphs, axis 1 are the estimators that selected them")
##            matrix = np.hstack(
##                        [self.cheater.cheat_calc([graphs[z] for z in costs_ranked[:,i]]).reshape(-1,1)
##                        for i in range(costs_ranked.shape[1])]
##                )
##            print(matrix)
##            print (costs[costs_ranked[:,1],0])
##            print (costs[costs_ranked[:,0],0])
##
##            stuff = np.where(matrix == 0.0)
##            if len(stuff[0])>0:
##                from util import util
##                util.dumpfile(graphs[costs_ranked[stuff][0]],"gr")
##                print ("graph dumped")

   
    def duplicate_rm(self,graphs):
        graphs  = list(self._duplicate_rm(graphs))
        return graphs

    def _duplicate_rm(self,graphs):
        hashes =self.hash_vectorizer.vectorize(graphs)
        self.collisionlist =[]
        for i, (ha, gr) in enumerate(zip(hashes,graphs)):
            if ha not in self.seen_graphs:
                self.seen_graphs[ha] = i
                yield gr
            else:
                self.collisionlist.append((i,self.seen_graphs[ha]))



    def get_costs(self, graphs):
        timenow=time.time()
        costs = self.multiobj_est.decision_function(graphs)
        #logger.debug("costs: best dist: %f (%.2fs)" %  (np.min(costs[:,0]) ,time.time()-timenow))
        sort = np.argsort(costs,axis=0)
        nucol = np.argsort(sort,axis=0)
        current_val = -1
        current_rank = -1
        resdic ={}
        for i,e in enumerate(sort[:,2]):
            if costs[e,2] != current_val: # there is a new value
                current_val = costs[e,2]      # remember that 
                resdic[current_val] = i
        for i,e in enumerate(costs[:,2]):
            nucol[i,2] = resdic[e]
        costs = np.hstack((costs, np.sum(nucol,axis =1).reshape(-1,1)))
        logger.log(10, f"costs: best dist: {np.min(costs[:,0])} ({time.time()-timenow})")
        return costs

    def _get_neighbors(self, graph):
        current_graph_vector = csr_matrix(vertex_vec(graph, _decomposer).sum(axis=0))
##        neighs = list(self.grammar.neighbors(graph=graph)) #####
        neighs = list(self.grammar.neighbors(graph=graph, selectordata=[self.target_graph_vector,
                                                                        current_graph_vector,
                                                                        self.use_normalization,
                                                                        self.cipselector_k]))
        return neighs

    def _expand_neighbors(self, graphs):
        global _decomposer ##### Stupid hack but I dont know how else to allow lambda functions in multiprocessing
        _decomposer = self.decomposer #####
        if self.multiproc>1:
            with multiprocessing.Pool(self.multiproc) as p:
                res = list(concat(p.map(self._get_neighbors,graphs)))
                return res
        else:
            res = list(concat(map(self._get_neighbors,graphs)))
            return res


    def _get_score_substitution(self, graph): #
        """Only used with Cipselector Option 0. Replaces _get_neighbors"""
        current_graph_vector = csr_matrix(vertex_vec(graph, self.decomposer).sum(axis=0))
        neighs = list(self.grammar.scored_substitutions(graph=graph, selectordata=[self.target_graph_vector,
                                                                        current_graph_vector,
                                                                        self.use_normalization,
                                                                        self.cipselector_k]))
        return neighs

    def _expand_neighbors2(self, graphs): #
        """Only used with Cipselector Option 0. Replaces _expand_neighbors"""
        if self.multiproc>1:
            with multiprocessing.Pool(self.multiproc) as p:
                res = list(concat(p.map(self._get_score_substitution,graphs)))
        else:
            res = list(concat(map(self._get_score_substitution,graphs)))
        res.sort(reverse=True, key=lambda a: a[0])
        counter = 0
        grlist = []
        for sc,sub,gr in res:
            graph_ = self.grammar._substitute_core(gr, sub[0], sub[1])
            if graph_ is not None:
                grlist.append(graph_)
                counter += 1
                if counter >= self.cipselector_k:
                    break
        return grlist


class lsgg_size_hack(lsgg): # Back in use!
    def _neighbors_given_cips(self, graph, orig_cips):
        """iterator over graphs generted by substituting all orig_cips in graph (with
        cips from grammar)"""
        grlen = len(graph)
        for cip in orig_cips:
            congruent_cips = self._get_congruent_cips(cip) # Got renamed. (was "self._congruent_cips()")
            cips_ = [nucip for nucip in congruent_cips 
                        if (nucip.core_nodes_count + grlen - cip.core_nodes_count) <= self.genmaxsize]
            for cip_ in cips_:
                graph_ = self._substitute_core(graph, cip, cip_) # Got renamed. (was "self.core_substitution()")
                if graph_ is not None:
                    yield graph_

    def neighbors(self, graph, selectordata, filter = lambda x:True):
        """iterator over all neighbors of graph (that are conceiveable by the grammar)
        OVERWRITES the neighbors function of the cipcorevector LSGG"""
        grlen = len(graph)
        current_cips = self._get_cips(graph,filter)
        current_cips_congrus = [(current_cip,concip) for current_cip in current_cips 
                for concip in self._get_congruent_cips(current_cip) if len(concip.core_nodes) + grlen - len(current_cip.core_nodes) <= self.genmaxsize]
        filtered_current_other = self.cipselector(current_cips_congrus,*selectordata)
        for current_cip, congru in filtered_current_other:
            graph_ = self._substitute_core(graph, current_cip, congru)
            if graph_ is not None:
                yield graph_


    def scored_substitutions(self, graph, selectordata, filter = lambda x:True): #
        """Returns score subsitutions"""
        grlen = len(graph)
        current_cips = self._get_cips(graph,filter)
        current_cips_congrus = [(current_cip,concip) for current_cip in current_cips 
                for concip in self._get_congruent_cips(current_cip) if len(concip.core_nodes) + grlen - len(current_cip.core_nodes) <= self.genmaxsize]
        for score,production in new_cipselector0(current_cips_congrus,*selectordata):
            yield score, production, graph



class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            core_sizes=[0,1,2],
            context_size=2,
            min_count=1,
            expand_max_n_neighbors=None,
            max_size_frontier=None,
            n_iter=5,
            expand_max_frontier=1000,
            keepgraphs=30,
            output_k_best=None,
            add_grammar_rules = False,
            graph_size_limiter = lambda x: 999,
            squared_error = False,
            adapt_grammar_n_iter=None, cs2cs=[] , # context size 2 core size
            multiproc=False,
            decomposer=None,
            cipselector_option=None, **kwargs):
        """init."""
        if cipselector_option == 1:
            cipselector = new_cipselector
        elif cipselector_option == 2:
            cipselector = new_cipselector2
        elif cipselector_option == 0: # Special option
            cipselector = new_cipselector0
        else:
            raise ValueError("Invalid Cipselector Option")
        self.graph_size_limiter = graph_size_limiter
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.output_k_best = output_k_best
        self.decomposer = decomposer
        self.grammar = lsgg_size_hack(radii=core_sizes, thickness=context_size, core_vec_decomposer=decomposer, cipselector=cipselector, nodelevel_radius_and_thickness=True) #cip_root_all=False, half_step_distance=True)
###        self.grammar = lsggold(nodelevel_radius_and_thickness=True) #cip_root_all=False, half_step_distance=True)
##        self.grammar.radii = core_sizes #self.grammar.set_core_size(core_sizes)
##        self.grammar.thickness = context_size #self.grammar.decomposition_args['thickness_list'] = [context_size]
        #self.grammar.set_min_count(min_count) interfacecount 1 makes no sense
        self.grammar.filter_min_cip = min_count #self.grammar.filter_args['min_cip_count'] = min_count
        self.optiopts = kwargs
        self.cs2cs = cs2cs
        
        
        if multiproc == -1:
            self.usecpus = multiprocessing.cpu_count()
        elif multiproc ==0:
            self.usecpus = 1
        else:
            self.usecpus=multiproc

        self.multiobj_est = costs.DistRankSizeCostEstimator(r=r, d=d, multiproc=self.usecpus, squared_error=squared_error)
        self.add_grammar_rules = add_grammar_rules
        self.keepgraphs = keepgraphs

    def fit(self):
        """fit."""
        pass

    def enhance_grammar(self, graphs):

        if self.cs2cs:
            assert False, "Not adapted to new grammar."
            # train on context 1 +2  what is allowed
            self.grammar.decomposition_args['thickness_list'] = [2,4]
            cs = self.grammar.decomposition_args['radius_list']
            self.grammar.fit(graphs)

            # train on context 2
            self.grammar.decomposition_args['thickness_list'] = [4]
            self.grammar.set_core_size(self.cs2cs)
            self.grammar.fit(graphs)

            # set tings up for extraction
            self.grammar.decomposition_args['thickness_list'] = [2,4]
            self.grammar.set_core_size( cs+self.cs2cs )

        else: # fir normaly
            #from util.util import loadfile 
            #graphs = loadfile('.tasks')[0][:-50]
            self.grammar.fit(graphs)

        if self.add_grammar_rules:
            print(self.grammar)
            print ('enhance grammar is not to be used anymore... 1 should be appended when not already in.. ')
            self.grammar.decomposition_args['thickness_list'].append(1)
            lsggs.enhance(self.grammar, graphs,lsggs.makelsgg(),nodecount=10, edgecount =5, degree =3)
            print(self.grammar)
        
        #from graphlearn3 import util
        #util.draw_grammar_term(self.grammar.productions)
        '''
        logger.debug("before lsgg enhancement: "+str(self.grammar))
        self.grammar = lsggscramble.enhance(self.grammar,
                                            graphs[:20],
                                            makelsgg=lsggscramble.makelsgg,
                                            nodecount =11,
                                            edgecount =5,
                                            degree =3)
        self.grammar._is_fit = True
        logger.debug("after lsgg enhancement: "+str(self.grammar))
        '''

    def optimize(
            self,
            reference_graphs,
            desired_distances,
            ranked_graphs, start_graph_list=False, 
            target_graph_vector=None, target=None):
        """optimize.
        # providing target, prints real distances for all "passing" creations
        """
        self.target_graph_vector = target_graph_vector
        # self.grammar.fit(graphscramble.scramble(ranked_graphs))
        self.enhance_grammar(ranked_graphs)  # WAS REF BUT REF MIGHT BE SMALL

        # fit objectives
        self.multiobj_est.fit(desired_distances,
                              reference_graphs,
                              ranked_graphs)

        self.grammar.genmaxsize = self.calc_graph_max_size(reference_graphs)
        # setup and run optimizer

        pgo = MYOPTIMIZER(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            n_iter=self.n_iter,
            keepgraphs=self.keepgraphs,
            multiproc =self.usecpus,
            target_graph_vector=target_graph_vector,
            target=target, decomposer=self.decomposer,**self.optiopts)

        if not start_graph_list:
            res = pgo.optimize(ranked_graphs)
            #res = pgo.optimize(reference_graphs + ranked_graphs)
        else:
            res = pgo.optimize(start_graph_list)
        return res


    def calc_graph_max_size(self,graphs):
        graphlengths = np.array([len(g)+g.number_of_edges() for g in graphs])
        val  = self.graph_size_limiter(graphlengths)
        logger.debug("debug values for size cutoff calculation")
        logger.debug(val)
        logger.debug(graphlengths)
        return val
        


