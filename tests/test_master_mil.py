#!/usr/bin/env python

"""Tests for `master_mil` package."""
import logging

import pytest

from master_mil.classification import train_bag_classifier, evaluate_model
from master_mil.cli import main_function
from master_mil.simulate import SimpleMILDistribution


def test_acceptance():
    """Sample pytest test function with the pytest fixture as an argument."""
    logging.basicConfig(level=logging.INFO)
    witness_rate = 0.5
    simulator = SimpleMILDistribution(0, 1, witness_rate=witness_rate)
    data = simulator.sample(100)
    classifier = train_bag_classifier(data)
    accuracy = evaluate_model(classifier, data)
    print(f'Witness rate {witness_rate} accuracy {accuracy}')



    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
