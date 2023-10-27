"""Console script for master_mil."""
# todo


import typer

from master_mil.classification import evaluate_model, train_bag_classifier
from master_mil.simulate import SimpleMILDistribution


def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main()

    '''
    for witness_rate in [0.05, 0.1, 0.2, 0.5]:
        simulator = SimpleMILDistribution(0, 1, witness_rate=witness_rate)
        data = simulator.sample(100)
        classifier = train_bag_classifier(data)
        accuracy = evaluate_model(classifier, data)
        print(f'Witness rate {witness_rate} accuracy {accuracy}')


def main():
    typer.run(main_function)


if __name__ == "__main__":
    main()
