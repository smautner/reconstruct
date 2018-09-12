import json

import dill
import structout as so
from eden.graph import Vectorizer
from sklearn.neighbors import NearestNeighbors

dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
loadfile = lambda filename: dill.load(open(filename, "rb"))
jloadfile = lambda filename:  json.loads(open(filename,'w').read())


class InstanceMaker(object):
    """InstanceMaker."""

    def __init__(self, n_landmarks=5, n_neighbors=50):
        """init."""
        self.vec = Vectorizer(r=3, d=6,normalization=False, inner_normalization=False)
        self.n_neighbors = n_neighbors
        self.n_landmarks = n_landmarks

    def fit(self, graphs, ntargets):
        """graphs/targets split, trains NN on graphs"""
        self.graphs =  graphs[:-ntargets]
        self.targets = graphs[-ntargets:]

        vecs = self.vec.transform(self.graphs)
        if self.n_neighbors > len(self.graphs):
            self.n_neighbors = len(self.graphs)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(vecs)
        return self


    def get(self):
        target_graph = self.targets.pop()
        target_vec = self.vec.transform([target_graph])
        distances, neighbors = self.nn.kneighbors(target_vec, return_distance=True)
        distances = distances[0]
        neighbors = neighbors[0]
        ranked_graphs = [self.graphs[i] for i in neighbors]
        landmark_graphs = ranked_graphs[:self.n_landmarks]
        desired_distances = distances[:self.n_landmarks]

        print ("target and nn")
        so.gprint([target_graph, ranked_graphs[0]], edgelabel='label')
        return landmark_graphs, desired_distances, ranked_graphs, target_graph