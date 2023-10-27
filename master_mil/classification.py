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
    N = NormalDistribution
    negative_bags = training_observations[training_observations.y == 0]
    mu_0 = np.mean(negative_bags.x)
    sigma_0 = np.std(negative_bags.x)
    n_iter = 1000
    positive_bags = training_observations[training_observations.y == 1]
    witness_rate = 0.5
    X = positive_bags.x
    mu_1 = np.max(X) # 1#  mu_0+ (np.mean(X)-mu_0) / witness_rate

    sigma_1 = sigma_0.copy()
    w = witness_rate
    log_likelihood = -np.inf
    for i in range(n_iter):
        # print(f'Iteration {i}: mu_1={mu_1}, sigma_1={sigma_1}, w={w}, log_likelihood={log_likelihood}')
        likelihood_1 = N(mu_1, sigma_1).log_prob(X)
        likelihood_0 = N(mu_0, sigma_0).log_prob(X)
        likelihoods = [np.log(w)+likelihood_0, np.log(1-w)+likelihood_1]
        log_K = np.logaddexp(*likelihoods)
        p_negative, p_positive = (np.exp(l-log_K) for l in likelihoods)
        p_positive = p_positive +  1e-10
        factor = np.sum(p_positive)
        mu_1 = np.sum(p_positive*X)/factor
        sigma_1 = np.sqrt(np.sum(p_positive*(X-mu_1)**2)/factor)
        w = np.mean(p_positive)-(1e-10/2)
        w = np.clip(w, 0.00, 1)
        positive_bag_model = MixtureModel([N(mu_1, sigma_1), N(mu_0, sigma_0)], [w, 1-w])
        log_likelihood = positive_bag_model.log_prob(X).sum()
        #assert log_likelihood > prev_log_likelihood, (log_likelihood, prev_log_likelihood)
        # prev_log_likelihood = log_likelihood

    negative_model = N(mu_0, sigma_0)
    return EMModel(positive_bag_model, negative_model)

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




