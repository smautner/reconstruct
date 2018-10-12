import logging
import time
import random
import queue
import heapq
import numpy as np
from eden.util import timeit
from graphlearn3.lsgg import lsgg
from toolz.curried import compose, map, concat

from exploration.pareto_funcs import _manage_int_or_float

logger = logging.getLogger(__name__)

import structout as so
from exploration import pareto_funcs as paretof, cost_estimator as costs
from extensions import lsggscramble as lsggs

class ParetoGraphOptimizer(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            grammar=None,
            multiobj_est=None,
            expand_max_n_neighbors=None,
            n_iter=19,
            expand_max_frontier=20,
            max_size_frontier=30,
            adapt_grammar_n_iter=None,
            random_state=1):

        """init."""
        self.grammar = grammar
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.multiobj_est = multiobj_est
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.pareto_set = dict()
        self.knn_ref_dist = None

        self.curr_iter = 0
        self.prev_costs = None

        random.seed(random_state)

        '''
        self._expand_neighbors = compose(list,
                                         concat,
          map(self._get_neighbors))
        '''                               

    def _mark_non_visited(self, graphs):
        for g in graphs:
            g.graph['visited'] = False

    @timeit
    def optimize(self, graphs):
        """Optimize iteratively."""
        assert (self.grammar.is_fit())
        assert (self.multiobj_est.is_fit())
        # init
        costs = self.multiobj_est.decision_function(graphs)
        seed_graphs = paretof.get_pareto_set(graphs, costs)
        self.pareto_set = seed_graphs
        self._mark_non_visited(self.pareto_set)

        # main cycle
        try:
            for i in range(self.n_iter):
                seed_graphs = self._update_pareto_set(seed_graphs)
        finally:
            return seed_graphs
        '''
        try:
            last(islice(
                iterate(self._update_pareto_set, seed_graphs),
                self.n_iter))
        except Exception as e:
            msg = 'Terminated at iteration:%d because %s' % (self.curr_iter, e)
            logger.debug(traceback.format_exc())
            logger.debug(msg)
        finally:
            return self.pareto_set
        '''

    def _get_neighbors(self, graph):
        n = self.expand_max_n_neighbors
        if n is None:
            g = list(self.grammar.neighbors(graph))
            # logger.debug("generated %d neighbours" % len(g))
            # print "ja lol ich bin da", len(g)
            # if len(g) == 0:
            # draw.graphlearn(graph, edge_label='label',secondary_vertex_label='id')

            # print 'using grammar', self.grammar
            # cips = list(self.grammar._cip_extraction(graph))
            # map(draw.decorate_cip, cips)
            # draw.graphlearn([c.graph for c in cips], title_key='title')
            # for e in cips:
            #    print 'g hits for %d %d' % (e.interface_hash, len(self.grammar.productions.get(e.interface_hash,[])))

            # exit()
            return g
        else:
            return self.grammar.neighbors_sample(graph, n_neighbors=n)

    def _update_grammar_policy(self):
        if self.curr_iter > 0 and \
                self.adapt_grammar_n_iter is not None and \
                self.curr_iter % self.adapt_grammar_n_iter == 0:
            logger.debug("throwing away grammar")
            min_count = self.grammar.get_min_count()
            min_count = min_count + 1
            self.grammar.set_min_count(min_count)
            self.grammar.reset_productions()
            self.grammar.fit(self.pareto_set)

            self._mark_non_visited(self.pareto_set)
            logger.debug(self.grammar)

    def _update_pareto_set_policy(self, neighbor_graphs):
        graphs = self.pareto_set + neighbor_graphs
        costs = self.multiobj_est.decision_function(graphs)
        self.pareto_set = paretof.get_pareto_set(graphs, costs)
        if self.max_size_frontier is not None:
            # reduce Pareto set size by taking best ranking subset
            m = self.max_size_frontier
            n = len(self.pareto_set)
            size = _manage_int_or_float(m, n)
            self.pareto_set = self.multiobj_est.select(self.pareto_set, size)
        return costs

    def _update_pareto_set_expansion_policy(self):
        size = _manage_int_or_float(self.expand_max_frontier,
                                    len(self.pareto_set))
        # permute the elements in the frontier
        ids = np.arange(len(self.pareto_set))
        np.random.shuffle(ids)
        ids = list(ids)
        # select non visited elements
        is_visited = lambda g: g.graph.get('visited', False)
        non_visited_ids = [id for id in ids
                           if not is_visited(self.pareto_set[id])]
        if len(non_visited_ids) == 0:
            raise Exception('No non visited elements in frontier, stopping')
        not_yet_visited_graphs = []
        for id in non_visited_ids[:size]:
            self.pareto_set[id].graph['visited'] = True
            not_yet_visited_graphs.append(self.pareto_set[id])
        return not_yet_visited_graphs

    def _update_pareto_set(self, seed_graphs):
        """_update_pareto_set."""
        # logger.debug("neighs of 1st seed:")
        # logger.debug(so.graph.make_picture(seed_graphs[0], edgelabel='label'))
        # logger.debug(so.graph.make_picture(self._expand_neighbors([seed_graphs[0]])[:10],
        #                                   edgelabel='label'))
        # self._update_grammar_policy()
        neighbor_graphs = self._expand_neighbors(seed_graphs)
        costs = self._update_pareto_set_policy(neighbor_graphs)
        new_seeds = self._update_pareto_set_expansion_policy()
        self._log_update_pareto_set(
            costs, self.pareto_set, neighbor_graphs, new_seeds)

        self.curr_iter += 1
        return new_seeds

    def _log_update_pareto_set(self,
                               costs,
                               pareto_set,
                               neighbor_graphs,
                               new_seed_graphs):
        # logger.debug("graphs in pareto step:")
        # logger.debug(so.graph.make_picture(pareto_set, edgelabel='label'))
        min_cost0 = min(costs[:, 0])
        par_size = len(pareto_set)
        med_cost0 = np.percentile(costs[:, 0], 50)
        txt = 'iter: %3d \t' % self.curr_iter
        txt += 'current min obj-0: %7.2f \t' % min_cost0
        txt += 'median obj-0: %7.2f \t' % med_cost0
        txt += 'added n neighbors: %4d \t' % len(neighbor_graphs)
        txt += 'obtained pareto set of size: %4d \t' % par_size
        txt += 'next round seeds: %4d ' % len(new_seed_graphs)
        logger.debug(txt)


import eden.graph
import itertools
import multiprocessing

import graphlearn3.lsgg_cip as glcip

class hashvec(object):

    def __init__(self, vectorizer, multiproc = True):
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
        hashed_features = [ hash(tuple(sorted(fd.keys()))) for fd in feature_dicts]
        return hashed_features
    
    def vectorize_chunk_glhash(self,chunk):
        return [glcip.graph_hash(eden.graph._edge_to_vertex_transform(g),2**20-1,node_name_label=lambda id,node:hash(node['label'])) for g in chunk]

    def vectorize_multiproc(self, graphs):
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            res = (p.map(self.vectorize_chunk, self.grouper(1000,graphs)))
        return itertools.chain.from_iterable(res)

    def vectorize(self,graphs):
        if self.multiproc:
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
            keeptop= 20,
            random_state=1,multiproc=True, target=None):
        """init."""
        self.grammar = grammar
        self.keeptop = keeptop
        self.multiobj_est = multiobj_est
        self.n_iter = n_iter
        random.seed(random_state)
        self.multiproc = multiproc
        self.hash_vectorizer = hashvec(eden.graph.Vectorizer(normalization=False,r=2,d=2),multiproc=multiproc)
        self.cheat = False
        self.seen_graphs = {}
        self.queues  = [ queue.PriorityQueue(maxsize=200) for i in range(3)]
        if target:
            self.cheat= True
            self.cheater = cheater(target)



    def checkstatus(self,costs,graphs):
        g = [graphs[e] for e in np.argmin(costs,axis=0)]
        #logger.debug([len(e)for e in g])
        logger.debug(so.graph.make_picture(g,edgelabel='label',size=10))
        if min(costs[:,0]) == 0:
            #logger.debug(graphs[np.argmin(costs[:,0])].graph['history'])
            return True
        return False

    @timeit
    def optimize(self, graphs):
        """Optimize iteratively."""
        assert (self.grammar.is_fit())
        assert (self.multiobj_est.is_fit())

        # init
        self.seen_graphs = {}
        for g in graphs:
            g.graph['history']=[]

        # main cycle
        for i in range(self.n_iter):
            graphs, status= self.optimize_step(graphs)
            if status:
                return True,i
            if not graphs:
                logger.debug("ran out of graphs")
                return False,i

        costs = self.get_costs(graphs)
        return self.checkstatus(costs, graphs),i

    def optimize_step(self, graphs):
        # filter, expand, chk duplicates
        logger.debug("++++++++  NEW OPTI ROUND +++++++")
        costs = self.get_costs(graphs)
        status = self.checkstatus(costs, graphs)
        graphs = self.filter_by_cost(costs, graphs)
        graphs = self._expand_neighbors(graphs)
        graphs = self.duplicate_rm(graphs)
        return graphs, status

    def filter_by_cost3(self,costs,graphs):
        """the idea is to use priority queues..."""
        timenow=time.time()
        in_count = len(graphs)
        if in_count< 50:
            logger.debug('cost_filter: keep all %d graphs' % in_count)
            return graphs

        # put graphs in queues
        for q,v,g in zip(self.queues,[costs[i,:] for i in range(3)],graphs):
            q.put((v,g))

        res = [q.get() for q in self.queues for i in range(10)]

        # DEBUG TO SHOW THE REAL DISTANCE
        if self.cheat:
            print ("real distances for all kept graphs, axis 1 are the estimators that selected them")
            matrix = np.hstack(
                        [self.cheater.cheat_calc([graphs[z] for z in costs_ranked[:,i]]).reshape(-1,1)
                        for i in range(costs_ranked.shape[1])]
                )
            print(matrix)
            print (costs[costs_ranked[:,1],0])
            print (costs[costs_ranked[:,0],0])

            stuff = np.where(matrix == 0.0)
            if len(stuff[0])>0:
                from util import util
                util.dumpfile(graphs[costs_ranked[stuff][0]],"gr")
                print ("graph dumped")

        logger.debug('cost_filter: got %d graphs, reduced to %d (%.2fs)'%(in_count,len(res),time.time()-timenow))
        return res
    def filter_by_cost(self,costs,graphs):
        """expand top 10 in everything, discard rest"""
        timenow=time.time()
        in_count = len(graphs)
        if in_count< 50:
            logger.debug('cost_filter: keep all %d graphs' % in_count)
            return graphs
       
        # need to keep x best in each category
        costs_ranked = np.argsort(costs,axis=0)[:self.keeptop]
        want , counts = np.unique(costs_ranked,return_counts=True) 

        res = [graphs[idd] for idd,count in zip( want,counts) if count > 0 ] 
        for g,score in zip(res,want):
            g.graph['history'].append(costs[score])

        # DEBUG TO SHOW THE REAL DISTANCE
        if self.cheat:
            print ("real distances for all kept graphs, axis 1 are the estimators that selected them")
            matrix = np.hstack(
                        [self.cheater.cheat_calc([graphs[z] for z in costs_ranked[:,i]]).reshape(-1,1)
                        for i in range(costs_ranked.shape[1])]
                )
            print(matrix)
            print (costs[costs_ranked[:,1],0])
            print (costs[costs_ranked[:,0],0])

            stuff = np.where(matrix == 0.0)
            if len(stuff[0])>0:
                from util import util
                util.dumpfile(graphs[costs_ranked[stuff][0]],"gr")
                print ("graph dumped")

        logger.debug('cost_filter: got %d graphs, reduced to %d (%.2fs)'%(in_count,len(res),time.time()-timenow))
        return res

    def filter_by_cost2(self,costs,graphs):
        """like 1, but prefilters by the size estimator, this does not works well if size is known..."""
        timenow=time.time()
        in_count = len(graphs)
        if in_count< 50:
            logger.debug('cost_filter: keep all %d graphs' % in_count)
            return graphs

        # 1. Filter by SIZE
        costs_ranked = np.argsort(costs,axis=0)[:,2]
        cut = 49
        while costs[costs_ranked[cut-1],2] == costs[costs_ranked[cut],2] and cut+1 < len(graphs):
            cut+=1

        costs_ranked = costs_ranked[:cut+1] 
        costs = costs[costs_ranked,:] 
        graphs =[graphs[idd] for idd in costs_ranked]
       
        # need to keep x best in each category
        costs_ranked = np.argsort(costs,axis=0)[:self.keeptop]
        want , counts = np.unique(costs_ranked,return_counts=True) 

        res = [graphs[idd] for idd,count in zip( want,counts) if count > 0 ] 
        for g,score in zip(res,want):
            g.graph['history'].append(costs[score])

        # DEBUG TO SHOW THE REAL DISTANCE
        if self.cheat:
            print ("real distances for all kept graphs, axis 1 are the estimators that selected them")
            matrix = np.hstack(
                        [self.cheater.cheat_calc([graphs[z] for z in costs_ranked[:,i]]).reshape(-1,1)
                        for i in range(costs_ranked.shape[1])]
                )
            print(matrix)
            print (costs[costs_ranked[:,1],0])
            print (costs[costs_ranked[:,0],0])

            stuff = np.where(matrix == 0.0)
            if len(stuff[0])>0:
                from util import util
                util.dumpfile(graphs[costs_ranked[stuff][0]],"gr")
                print ("graph dumped")

        logger.debug('cost_filter: got %d graphs, reduced to %d (%.2fs)'%(in_count,len(res),time.time()-timenow))
        return res

    def duplicate_rm(self,graphs):
        timenow=time.time()
        count = len(graphs)
        graphs  = list(self._duplicate_rm(graphs))
        logger.debug("duplicate_rm: %d -> %d graphs (%.2fs)" % (count, len(graphs), time.time()-timenow))
        return graphs

    def _duplicate_rm(self,graphs):
        hashes =self.hash_vectorizer.vectorize(graphs)
        for ha, gr in zip(hashes,graphs):
            if ha not in self.seen_graphs:
                self.seen_graphs[ha] =1
                yield gr



    def get_costs(self, graphs):
        timenow=time.time()
        costs = self.multiobj_est.decision_function(graphs)


        #logger.debug("costs: best dist: %f (%.2fs)" %  (np.min(costs[:,0]) ,time.time()-timenow))
        #return costs

        if costs.shape[0] < 50:
            return costs


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

        logger.debug("costs: best dist: %f (%.2fs)" %  (np.min(costs[:,0]) ,time.time()-timenow))
        return costs

    def _get_neighbors(self, graph):
        neighs = list(self.grammar.neighbors(graph))
        for n in neighs:
            n.graph['history']= graph.graph['history'].copy()
        return neighs

    def _expand_neighbors(self, graphs):
        if self.multiproc:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                return list(concat(p.map(self._get_neighbors,graphs)))
        else:
          return list(concat(map(self._get_neighbors,graphs)))

class lsgg_size_hack(lsgg):
    def _neighbors_given_cips(self, graph, orig_cips):
        """iterator over graphs generted by substituting all orig_cips in graph (with cips from grammar)"""
        grlen = len(graph)
        for cip in orig_cips:
            cips_ = self._congruent_cips(cip)
            cips_ = [c for c in cips_ if c.core_nodes_count+grlen <= self.genmaxsize]
            for cip_ in cips_:
                graph_ = self._core_substitution(graph, cip, cip_)
                if graph_ is not None:
                    yield graph_

class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            half_step_distance=False,
            context_size=1,
            min_count=1,
            expand_max_n_neighbors=None,
            max_size_frontier=None,
            n_iter=5,
            expand_max_frontier=1000,
            keeptop= 20,
            output_k_best=None,
            add_grammar_rules = False,
            adapt_grammar_n_iter=None, multiproc=False):
        """init."""
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.output_k_best = output_k_best
        self.grammar = lsgg_size_hack(cip_root_all=False, half_step_distance=half_step_distance)
        self.grammar.set_core_size([0, 1, 2])
        if half_step_distance:
            self.grammar.set_core_size([0, 1, 2,3,4])
        self.grammar.set_context(context_size)
        #self.grammar.set_min_count(min_count) interfacecount 1 makes no sense
        self.grammar.filter_args['min_cip_count'] = min_count
        self.multiobj_est = costs.DistRankSizeCostEstimator(r=r, d=d, multiproc=multiproc)
        self.multiproc=multiproc
        self.add_grammar_rules = add_grammar_rules
        self.keeptop =keeptop

    def fit(self):
        """fit."""
        pass

    def enhance_grammar(self, graphs):
        self.grammar.fit(graphs)
        if self.add_grammar_rules:
            print(self.grammar)
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
            ranked_graphs, start_graph_list=False, target=None):
        """optimize.
        # providing target, prints real distances for all "passing" creations
        """
        # self.grammar.fit(graphscramble.scramble(ranked_graphs))
        self.enhance_grammar(ranked_graphs)  # WAS REF BUT REF MIGHT BE SMALL

        # fit objectives
        self.multiobj_est.fit(desired_distances,
                              reference_graphs,
                              ranked_graphs)

        grammar.genmaxsize = self.calc_graph_max_size(reference_graphs)
        # setup and run optimizer
        pgo = ParetoGraphOptimizer(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            expand_max_n_neighbors=self.expand_max_n_neighbors,
            expand_max_frontier=self.expand_max_frontier,
            max_size_frontier=self.max_size_frontier,
            n_iter=self.n_iter,
            adapt_grammar_n_iter=self.adapt_grammar_n_iter)

        pgo = MYOPTIMIZER(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            n_iter=self.n_iter,
            keeptop=self.keeptop,
            multiproc=self.multiproc, target=target)

        if not start_graph_list:
            res = pgo.optimize(ranked_graphs)
            #res = pgo.optimize(reference_graphs + ranked_graphs)
        else:
            res = pgo.optimize(start_graph_list)
        return res

    def calc_graph_max_size(self,graphs):
        graphlengths = np.array([len(g)+g.number_of_edges() for g in graphs])
        return graphlengths.max() + graphlengths.std()

# USAGE:
'''
    ld_opt = LocalLandmarksDistanceOptimizer(
        r=r,
        d=d,
        min_count=min_count,
        context_size=context_size,
        expand_max_n_neighbors=expand_max_n_neighbors,
        n_iter=n_iter + 1,
        expand_max_frontier=expand_max_frontier,
        max_size_frontier=max_size_frontier,
        output_k_best=output_k_best,
        adapt_grammar_n_iter=adapt_grammar_n_iter)

    graphs = ld_opt.optimize(
        landmark_graphs,
        desired_distances,
        ranked_graphs)
'''
