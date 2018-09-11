import logging
import time
import traceback
import random
from itertools import islice

import numpy as np
from eden.util import timeit
from graphlearn3.lsgg import lsgg
import graphlearn3.util as util
from toolz.curried import compose, map, concat
from toolz.itertoolz import iterate, last

from pareto_funcs import _manage_int_or_float

logger = logging.getLogger(__name__)

import lsggscramble
import structout as so
import cost_estimator as costs
import pareto_funcs as paretof

from cost_estimator import pvectorize

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

        self._expand_neighbors = compose(list,
                                         concat,
                                         map(self._get_neighbors))

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
import scipy

import graphlearn3.lsgg_cip as glcip
class hashvec(object):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

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

    def vectorize(self, graphs):
        
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            res = (p.map(self.vectorize_chunk, self.grouper(1000,graphs)))
        return itertools.chain.from_iterable(res)


class MYOPTIMIZER(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            grammar=None,
            multiobj_est=None,
            n_iter=19,
            random_state=1):
        """init."""
        self.grammar = grammar
        self.multiobj_est = multiobj_est
        self.n_iter = n_iter
        random.seed(random_state)
        self.hash_vectorizer = hashvec(eden.graph.Vectorizer(normalization=False,r=2,d=2))
        self._expand_neighbors = compose(list,
                                         concat,
                                         map(self._get_neighbors))

    def checkstatus(self,costs,graphs):
        g = [graphs[e] for e in np.argmax(costs,axis=0)]
        #logger.debug([len(e)for e in g])
        logger.debug(so.graph.make_picture(g,edgelabel='label',size=10))
        if min(costs[:,0]) == 0:
            print(graphs[np.argmin(costs[:,0])].graph['history'])
            exit()
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
            graphs = self.optimize_step(graphs)


        logger.debug("DONE")
        costs = self.get_costs(graphs)
        self.checkstatus(costs, graphs)

    def optimize_step(self, graphs):
        # filter, expand, chk duplicates
        logger.debug("++++++++  NEW OPTI ROUND +++++++")
        costs = self.get_costs(graphs)
        self.checkstatus(costs, graphs)
        graphs = self.filter_by_cost(costs, graphs)
        graphs = self._expand_neighbors(graphs)
        graphs = self.duplicate_rm(graphs)
        return graphs

    def filter_by_cost(self,costs,graphs):
        timenow=time.time()
        in_count = len(graphs)
        if in_count< 40:
            logger.debug('cost_filter: keep all %d graphs' % in_count)
            return graphs
        costs_ranked = np.argsort(costs,axis=0) # column wise rank
        best_scores = np.sum( costs_ranked < 70, axis=1) # make 0/1 matrix of where 1 = graph who has an ok rank
        for g,scores in zip(graphs,costs_ranked < 70 ):
            g.graph['history'].append(scores)
        res = [g for g,good_if_true in zip(graphs,best_scores) if good_if_true > 1] # select the graphs

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
        if costs.shape[0] < 40:
            return costs
        costs_partitioned = np.partition(costs, -10, axis=0)  # find the 10th largest value column wise
        normalized_costs = costs / costs_partitioned[-10,:]  # divide by that
        costs = np.hstack((costs, np.sum(normalized_costs, axis=1).reshape(-1,1)))
        logger.debug("costs: best dist: %f (%.2fs)" %  (np.min(costs[:,0]) ,time.time()-timenow))

        return costs




    def _get_neighbors(self, graph):
        neighs = list(self.grammar.neighbors(graph))
        for n in neighs:
            n.graph['history']= graph.graph['history'].copy()
        return neighs


class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            context_size=2,
            min_count=1,
            expand_max_n_neighbors=None,
            max_size_frontier=None,
            n_iter=2,
            expand_max_frontier=1000,
            output_k_best=None,
            adapt_grammar_n_iter=None):
        """init."""
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.output_k_best = output_k_best
        self.grammar = lsgg(cip_root_all=False, half_step_distance=True)
        self.grammar.set_core_size([0, 1, 2])
        self.grammar.set_context([1])
        self.grammar.set_context(context_size)
        self.grammar.set_min_count(min_count)
        self.multiobj_est = costs.DistRankSizeCostEstimator(r=r, d=d)

    def fit(self):
        """fit."""
        pass

    def enhance_grammar(self, graphs):
        self.grammar.fit(graphs)
        # util.draw_grammar_term(self.grammar.productions)
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
            ranked_graphs, start_graph_list=False):
        """optimize."""
        # self.grammar.fit(graphscramble.scramble(ranked_graphs))
        self.enhance_grammar(ranked_graphs)  # WAS REF BUT REF MIGHT BE SMALL

        # fit objectives
        self.multiobj_est.fit(desired_distances,
                              reference_graphs,
                              ranked_graphs)

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
            n_iter=self.n_iter)

        if not start_graph_list:
            graphs = pgo.optimize(reference_graphs + ranked_graphs)
        else:
            graphs = pgo.optimize(start_graph_list)

        if self.output_k_best is None:
            return graphs
        else:
            # output a selection of the Pareto set
            return self.multiobj_est.select(
                graphs,
                k_best=self.output_k_best,
                objective=0)


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
