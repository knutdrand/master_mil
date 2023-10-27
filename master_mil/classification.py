import logging

import numpy as np
import scipy
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
import sklearn

from .datatypes import SimpleObservation
from typing import Protocol
logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


class SimpleModel:
    def __init__(self, logisitic_regression: LogisticRegression):
        self.logistic_regression = logisitic_regression

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logistic_regression.predict(x.max(axis=-1, keepdims=True))


def naive_train_bag_classifier(training_observations: SimpleObservation) -> Classifier:
    max_x = np.max(training_observations.x, axis=-1, keepdims=True)
    assert max_x.shape == (len(training_observations), 1)
    model = LogisticRegression()
    model.fit(max_x, training_observations.y)
    return SimpleModel(model)


def train_bag_classifier(training_observations: SimpleObservation) -> Classifier:
    negative_bags = training_observations[training_observations.y == 0]
    negative_distribution = NormalDistribution(np.mean(negative_bags.x), np.std(negative_bags.x))
    n_iter = 1000
    positive_bags = training_observations[training_observations.y == 1]
    X = positive_bags.x
    positive_distribution = NormalDistribution(np.max(X), negative_distribution.sigma)
    mixture_model = MixtureModel([positive_distribution, negative_distribution], [0.5, 0.5])
    for i in range(n_iter):
        positive_likelihood = positive_distribution.log_prob(X)
        total_likelihood = mixture_model.log_prob(X)
        p_positive = np.exp(positive_likelihood-total_likelihood)+1e-10
        factor = np.sum(p_positive)
        positive_distribution.mu = np.sum(p_positive*X)/factor
        positive_distribution.sigma = np.sqrt(np.sum(p_positive*(X-positive_distribution.mu)**2)/factor)
        w = np.mean(p_positive)-(1e-10/2)
        w = np.clip(w, 0.00, 1)
        mixture_model.weights = [w, 1-w]
    return EMModel(mixture_model, negative_distribution)


class EMModel:
    def __init__(self, positive_bag_model, negative_model):
        self.positive_bag_model = positive_bag_model
        self.negative_model = negative_model

    def predict(self, X):
        positive_likelihood = self.positive_bag_model.log_prob(X).sum(axis=-1)
        negative_likelihood = self.negative_model.log_prob(X).sum(axis=-1)
        return positive_likelihood > negative_likelihood


class NormalDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def prob(self, X):
        return scipy.stats.norm.pdf(X, self.mu, self.sigma)

    def log_prob(self, X):
        return scipy.stats.norm.logpdf(X, self.mu, self.sigma)


class MixtureModel:
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = np.asanyarray(weights)

    def log_prob(self, X):
        return logsumexp([np.log(w) + d.log_prob(X) for d, w in zip(self.distributions, self.weights)], axis=0)


def evaluate_model(model: Classifier, test_observations: SimpleObservation) -> float:
    predictions = model.predict(test_observations.x)
    return np.mean(predictions == test_observations.y)




