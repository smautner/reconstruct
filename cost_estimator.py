#!/usr/bin/env python
"""Provides cost estimators."""


import numpy as np
import scipy as sp
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from eden.graph import Vectorizer
from eden.util import describe
from eden.util import timeit
import logging
logger = logging.getLogger(__name__)
from scipy.stats import rankdata
from eden.graph import Vectorizer


class MultiObjectiveCostEstimator(object):
    """MultiObjectiveCostEstimator."""

    def __init__(self):
        """Initialize."""
        pass

    def set_params(self, estimators):
        """set_params."""
        self.estimators = estimators

    def decision_function(self, graphs):
        """decision_function."""
        cost_vec = [estimator.decision_function(graphs)
                    for estimator in self.estimators]
        costs = np.hstack(cost_vec)
        return costs

    def is_fit(self):
        """is_fit."""
        return self.estimators is not None

    def _compute_ranks(self, graphs):
        costs = self.decision_function(graphs)
        ranks = [rankdata(costs[:, i], method='min')
                 for i in range(costs.shape[1])]
        ranks = np.vstack(ranks).T
        return ranks

    def _select_avg_rank(self, ranks, k_best=10):
        agg_ranks = np.sum(ranks, axis=1)
        ids = np.argsort(agg_ranks)
        return ids[:k_best]

    def _select_single_objective(self, ranks, k_best=10, objective=None):
        agg_ranks = ranks[:, objective]
        ids = np.argsort(agg_ranks)
        if k_best == 1:
            return ids[0]
        else:
            return ids[:k_best]

    def _select_extremes(self, ranks):
        n_objectives = ranks.shape[1]
        ids = [self._select_single_objective(ranks, k_best=1, objective=i)
               for i in range(n_objectives)]
        ids = list(set(ids))
        return ids

    def select(self, graphs, k_best=10, objective=None):
        """select."""
        ranks = self._compute_ranks(graphs)
        if objective is None:
            ids = self._select_avg_rank(ranks, k_best)
            ext_ids = self._select_extremes(ranks)
            ids = list(set(list(ids) + list(ext_ids)))
        else:
            ids = self._select_single_objective(ranks, k_best, objective)
        k_best_graphs = [graphs[id] for id in ids]
        return k_best_graphs


class DistRankSizeCostEstimator(MultiObjectiveCostEstimator):
    """DistRankSizeCostEstimator."""

    def __init__(self, r=3, d=3):
        """Initialize."""
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)

    @timeit
    def fit(
            self,
            desired_distances,
            reference_graphs,
            ranked_graphs):
        """fit."""
        d_est = InstancesDistanceCostEstimator(self.vec)
        d_est.fit(desired_distances, reference_graphs)

        b_est = RankBiasCostEstimator(self.vec, improve=True)
        b_est.fit(ranked_graphs)

        s_est = SizeCostEstimator()
        s_est.fit(reference_graphs)

        self.estimators = [d_est, b_est, s_est]
        return self

class InstancesDistanceCostEstimator(object):
    """InstancesDistanceCostEstimator."""

    def __init__(self, vectorizer=Vectorizer()):
        """init."""
        self.desired_distances = None
        self.reference_vecs = None
        self.vectorizer = vectorizer

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        return self

    def _avg_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        dist_diff = (distances - d)
        avg_dist_diff = np.mean(np.absolute(dist_diff))
        return avg_dist_diff

    def decision_function(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        avg_distance_diff = np.array([self._avg_distance_diff(vec)
                                      for vec in x])
        avg_distance_diff = avg_distance_diff.reshape(-1, 1)
        return avg_distance_diff






class RankBiasCostEstimator(object):
    """RankBiasCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.estimator = SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True)
        self.improve = improve

    def fit(self, ranked_graphs):
        """fit."""
        x = self.vectorizer.transform(ranked_graphs)
        r, c = x.shape
        pos = []
        neg = []
        for i in range(r - 1):
            for j in range(i + 1, r):
                p = x[i] - x[j]
                n = - p
                pos.append(p)
                neg.append(n)
        y = np.array([1] * len(pos) + [-1] * len(neg))
        pos = sp.sparse.vstack(pos)
        neg = sp.sparse.vstack(neg)
        x_ranks = sp.sparse.vstack([pos, neg])
        logger.debug('fitting: %s' % describe(x_ranks))
        self.estimator = self.estimator.fit(x_ranks, y)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = np.absolute(scores)
        else:
            scores = - scores
        scores = scores.reshape(-1, 1)
        return scores


class SizeCostEstimator(object):
    """SizeCostEstimator."""

    def __init__(self):
        """init."""
        pass

    def _graph_size(self, g):
        return len(g.nodes()) + len(g.edges())

    def fit(self, graphs):
        """fit."""
        self.reference_size = np.percentile(
            [self._graph_size(g) for g in graphs], 50)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        sizes = np.array([self._graph_size(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        size_diffs = size_diffs.reshape(-1, 1)
        return size_diffs

