"""Console script for master_mil."""
import logging

# todo


import typer

from .classification import evaluate_model, train_bag_classifier, naive_train_bag_classifier
from .simulate import SimpleMILDistribution


def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main()

    '''
    for witness_rate in [0.05, 0.1, 0.2, 0.5]:
        simulator = SimpleMILDistribution(0, 1, witness_rate=witness_rate)
        train_data, test_data = (simulator.sample(100), simulator.sample(100))
        classifier = train_bag_classifier(train_data)
        accuracy = evaluate_model(classifier, test_data)
        naive_accuracy = evaluate_model(naive_train_bag_classifier(train_data), test_data)
        print(f'Witness rate {witness_rate} accuracy {accuracy}/{naive_accuracy}(naive)')


def main():
    typer.run(main_function)


if __name__ == "__main__":
    main()
