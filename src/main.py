import random

import params
import trainer
import tester


def get_data(class_params):
    """
    Method of generation names of test images for class
    :param class_params:
    :return: tuple of list of train and test data
    """
    train_data = set()
    test_data = set()
    while len(train_data) < class_params['train_count']:
        number = random.randint(1, class_params['count'])
        train_data.add(number)
    while len(test_data) < class_params['test_count']:
        number = random.randint(1, class_params['count'])
        if not number in train_data:
            test_data.add(number)
    get_name = lambda x: class_params["prefix"] + str(x) + class_params["postfix"]
    return map(get_name, train_data), map(get_name, test_data)


def train_and_test():
    """
    Simple implementation of train and test function
    """
    first_class_train_data, first_class_test_data = get_data(params.first_class_params)
    second_class_train_data, second_class_test_data = get_data(params.second_class_params)

    train_data = list(first_class_train_data) + list(second_class_train_data)
    train_response = [1 for i in xrange(0, len(first_class_train_data))] + [-1 for j in
                                                                            xrange(0, len(second_class_train_data))]
    solve_container = trainer.train(train_data, train_response)
    # solve_container.serialize(params.svm_save_file)

    test_data = list(first_class_test_data) + list(second_class_test_data)
    test_response = [1 for i in xrange(0, len(first_class_test_data))] + [-1 for j in
                                                                          xrange(0, len(second_class_test_data))]
    tester.test(test_data, test_response, solve_container)


import cProfile

cProfile.run("train_and_test()")

