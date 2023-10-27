import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn

from .datatypes import SimpleObservation
from typing import Protocol


class Classifier(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


class SimpleModel:
    def __init__(self, logisitic_regression: LogisticRegression):
        self.logistic_regression = logisitic_regression

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logistic_regression.predict(x.max(axis=-1, keepdims=True))


def train_bag_classifier(training_observations: SimpleObservation) -> Classifier:
    max_x = np.max(training_observations.x, axis=-1, keepdims=True)
    assert max_x.shape == (training_observations.x.shape[0], 1)
    model = LogisticRegression()
    model.fit(max_x, training_observations.y)
    return SimpleModel(model)


def evaluate_model(model: Classifier, test_observations: SimpleObservation) -> float:
    predictions = model.predict(test_observations.x)
    return np.mean(predictions == test_observations.y)




