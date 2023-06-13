import numpy as np
import itertools
from time import time
from scipy.optimize import minimize
from scipy.special import logsumexp


def construct_data(xs, ys):
    fill_value = np.zeros(xs[0].shape[1])
    xs = list(itertools.zip_longest(*xs, fillvalue=fill_value))
    xs = np.stack([np.stack([x[i] for x in xs]) for i in range(len(ys))])
    ys = np.column_stack(list(itertools.zip_longest(*ys, fillvalue=-1)))
    mask = (ys > -1).astype(int)
    return xs, ys, mask


class SequenceStorage:

    def __init__(self, num_features):
        self._data = {}
        self.num_features = num_features

    def __getitem__(self, item):
        assert isinstance(item, int)
        if item not in self._data:
            self._data[item] = np.array(list(itertools.product(*[list(range(self.num_features))] * item)))
        return self._data[item]


class CRF:

    def __init__(self, feature=None, transition=None, feature_shape=None, transition_shape=None):
        if feature is None and feature_shape is None:
            raise ValueError("Must have feature but both is none")
        if transition is None and transition_shape is None:
            raise ValueError("Must have transition but both is none")
        if feature is None:
            self.feature = np.zeros(feature_shape)
        else:
            self.feature = feature
        if transition is None:
            self.transition = np.zeros(transition_shape)
        else:
            self.transition = transition
        self.graph = []

    def log_feature_potentials(self, xs, ys=None):
        if ys is None:
            return xs.dot(self.feature.T)
        return np.sum(self.feature[ys.flatten()].reshape(ys.shape[0], ys.shape[1], -1) * xs, axis=2)

    def feature_potentials(self, xs, ys=None):
        return np.exp(self.log_feature_potentials(xs, ys))

    def log_transition_potentials(self, ys):
        y_shift0 = ys[:, :-1].flatten()
        y_shift1 = ys[:, 1:].flatten()
        return self.transition[y_shift0, y_shift1].reshape(ys.shape[0], -1)

    def transition_potentials(self, ys):
        return np.exp(self.log_transition_potentials(ys))

    def energy(self, xs, ys, mask=None):
        if mask is None:
            mask = np.ones(ys.shape)
        res = np.sum(self.log_feature_potentials(xs, ys) * mask, axis=1)
        res += np.sum(self.log_transition_potentials(ys) * mask[:, 1:], axis=1)
        return res

    def normalizing_constant(self, x, ys=None):
        if ys is None:
            ys = np.array(list(itertools.product(*[list(range(self.feature.shape[0]))] * len(x))))
        return logsumexp(self.energy(np.array([x]), ys))

    def log_normalizing_constant_mp(self, x, passing_cache=None):
        return logsumexp(self.__marginal_probability(x, passing_cache)[0])

    def seq_probs(self, x, ys=None):
        if ys is None:
            ys = np.array(list(itertools.product(*[list(range(self.feature.shape[0]))] * len(x))))
        xs = np.array([x])
        return np.exp(self.energy(xs, ys) - self.normalizing_constant(x, ys))

    def char_marginal_probs(self, x, ys=None):
        if ys is None:
            ys = np.array(list(itertools.product(*[list(range(self.feature.shape[0]))] * len(x))))
        probs = self.seq_probs(x, ys)
        res = np.zeros((len(x), self.feature.shape[0]))
        for y, p in zip(ys, probs):
            res[np.arange(len(x)), y] += p
        return res

    def log_message_passing(self, x, src, dst):
        return self.__message_passing(src, dst, self.log_feature_potentials(x[None, :])[0], {})

    def __message_passing(self, src, dst, log_feature_potentials, cache):
        scores = log_feature_potentials[src][np.newaxis, :] + self.transition
        for neighbor in self.graph[src]:
            if neighbor == dst:
                continue
            if (neighbor, src) in cache:
                scores += cache[(neighbor, src)]
            else:
                scores += self.__message_passing(neighbor, src, log_feature_potentials, cache)
        scores = logsumexp(scores, axis=1)
        cache[(src, dst)] = scores
        return scores

    def __marginal_probability(self, x, passing_cache=None):
        self.__construct_markov_graph(x)
        log_feature_potentials = self.log_feature_potentials(x[None, :])[0]
        cache = passing_cache or {}
        return np.array([
            np.sum([self.__message_passing(j, i, log_feature_potentials, cache) for j in self.graph[i]], axis=0)
            for i in range(len(x))
        ]) + log_feature_potentials

    def marginal_probability(self, x, passing_cache=None):
        score = self.__marginal_probability(x, passing_cache)
        return np.exp(score - logsumexp(score, axis=1)[:, None])

    def pairwise_marginal_probability(self, x, passing_cache=None):
        self.__construct_markov_graph(x)
        log_feature_potentials = self.log_feature_potentials(x[None, :])[0]
        scores = np.empty((len(x) - 1, len(self.feature), len(self.feature)))
        cache = passing_cache or {}
        for i in range(len(x) - 1):
            j = i + 1
            scores[i] = self.transition + log_feature_potentials[i][:, None] + log_feature_potentials[j]
            for k in [n for n in self.graph[i] if n != j]:
                scores[i] += self.__message_passing(k, i, log_feature_potentials, cache)[:, None]
            for l in [n for n in self.graph[j] if n != i]:
                scores[i] += self.__message_passing(l, j, log_feature_potentials, cache)
            scores[i] = np.exp(scores[i] - logsumexp(scores[i]))
        return scores

    def predict(self, xs: list):
        pred = []
        for x in xs:
            self.__construct_markov_graph(x)
            marginal = self.marginal_probability(x)
            pred.append(marginal.argmax(axis=1))
        return pred

    def log_likelihood(self, original_xs, xs, ys, mask=None):
        energy = self.energy(xs, ys, mask)
        z = np.mean([self.log_normalizing_constant_mp(x) for x in original_xs])
        return (energy - z).mean()

    def __construct_markov_graph(self, x):
        self.graph = [[1]] + [[i - 1, i + 1] for i in range(1, len(x) - 1)] + [[len(x) - 2]]

    def _feature_gradient(self, xs, ys):
        N = len(xs)
        C = len(self.feature)
        res = np.zeros((C, self.feature.shape[1]))
        for i in range(N):
            de = np.stack([(ys[i] == c) for c in range(C)])
            me = self.marginal_probability(xs[i], {}).T
            res += (de - me) @ xs[i]
        res /= N
        return res

    def _transition_gradient(self, xs, ys):
        N = len(xs)
        C = len(self.feature)
        res = np.zeros((C, C))
        for i in range(N):
            de = np.zeros((C, C))
            de[ys[i][:-1], ys[i][1:]] += 1
            me = self.pairwise_marginal_probability(xs[i], {}).sum(axis=0)
            res += de - me
        res /= N
        return res

    def flattened_gradient(self, xs, ys):
        return np.concatenate((self._feature_gradient(xs, ys).flatten(), self._transition_gradient(xs, ys).flatten()))

    def flattened_params(self):
        return np.concatenate((self.feature.flatten(), self.transition.flatten()))

    def from_weight(self, w):
        self.feature = w[:np.product(self.feature.shape)].reshape(self.feature.shape)
        self.transition = w[np.product(self.feature.shape):].reshape(self.transition.shape)
        return self

    def randomize_model(self):
        self.feature = np.random.randn(*self.feature.shape)
        self.transition = np.random.randn(*self.transition.shape)

    def train(self, xs, ys, randomize=False):
        if randomize:
            self.randomize_model()
        result = minimize(fun=objective, x0=np.concatenate((self.feature.flatten(), self.transition.flatten())),
                          args=(self, xs, ys), jac=gradient, method='L-BFGS-B')
        self.from_weight(result['x'])
        return self

    def train_profiler(self, xs, ys, randomize=True):
        t = time()
        return self.train(xs, ys, randomize), time() - t

    @staticmethod
    def from_data(xs, ys, randomize=False):
        num_labels = len(np.unique(np.array(ys)))
        crf = CRF(feature_shape=(num_labels, xs[0].shape[1]), transition_shape=(num_labels, num_labels))
        crf.train(xs, ys, randomize=randomize)
        return crf


def objective(w, *args):
    model, xs, ys = args
    return -model.from_weight(w).log_likelihood(xs, *construct_data(xs, ys))


def gradient(w, *args):
    model, xs, ys = args
    return -model.from_weight(w).flattened_gradient(xs, ys)
