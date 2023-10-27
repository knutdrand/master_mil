import numpy as np

from master_mil.datatypes import SimpleObservation


class SimpleMILDistribution:
    def __init__(self, mu_1, mu_2, sigma=1, witness_rate=0.1, bag_size=100):
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.sigma = sigma
        self.witness_rate = witness_rate
        self.bag_size = bag_size

    def sample(self, shape: int) -> SimpleObservation:
        """
        Sample shape bags of instances from the distribution.
        Positive bags are a mix of positive and negative instances,
        (self.witness_rate is the rate of positive instances),
        while negative bags are only negative instances.

        Parameters
        ----------
        shape

        Returns
        -------
        SimpleObservation where x is a (shape, bag_size) array of instances and y is a (shape,) array of labels.

        """
        y = np.random.randint(2, size=shape)
        z = np.random.binomial(1, self.witness_rate, size=(shape, self.bag_size))
        x = np.random.normal(np.where(y[:, None] & z, self.mu_2, self.mu_1),
                             self.sigma)
        assert x.shape == (shape, self.bag_size)
        return SimpleObservation(x, y)
