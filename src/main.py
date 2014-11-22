import random

import params
from feature_extractors.feature_extractor import FeatureExtractor
from trainer import Trainer
from tester import Tester
from image_loader.image_loader import ImageLoader


def get_class_data(class_params, sample_size):
    """

    :param class_params:
    :param sample_size:
    :return:
    """
    indexes = random.sample(range(0, class_params["image_count"]), sample_size)
    return map(
        lambda index: tuple([class_params["prefix"] + str(index) + class_params["postfix"], class_params["label"]]),
        indexes)


def get_train_and_test_data(class_params):
    """
Method of generation names of test images for class
:param class_params:
:return: tuple of list of train and test data
"""
    # train_data = set()
    # test_data = set()
    # while len(train_data) < class_params['train_count']:
    # number = random.randint(1, class_params['image_count'])
    # train_data.add(number)
    # while len(test_data) < class_params['test_count']:
    # number = random.randint(1, class_params['image_count'])
    #     if not number in train_data:
    #         test_data.add(number)
    # get_name = lambda x: tuple([class_params["prefix"] + str(x) + class_params["postfix"], class_params['label']])
    # return map(get_name, train_data), map(get_name, test_data)
    sample = get_class_data(class_params, class_params["train_count"] + class_params["test_count"])
    return Trainer.split_data(sample, [class_params["train_count"], class_params["test_count"]])


def run_customization(image_loader, feature_extractor):
    data = get_class_data(params.first_class_params, 5000) + get_class_data(params.second_class_params, 5000)
    random.shuffle(data)
    trainer = Trainer(image_loader, feature_extractor)
    c_range = [10 ** i for i in xrange(-5, 10)]
    gamma_range = [10 ** i for i in xrange(-5, 5)]
    results = trainer.svm_params_customization(data, params.svm_params, c_range, gamma_range)
    return results


def train_and_test(image_loader, feature_extractor):
    """
Simple implementation of train and test function
:param image_loader:
:param feature_extractor:
"""
    first_class_train_data, first_class_test_data = get_train_and_test_data(params.first_class_params)
    second_class_train_data, second_class_test_data = get_train_and_test_data(params.second_class_params)

    train_data = list(first_class_train_data) + list(second_class_train_data)
    random.shuffle(train_data)
    trainer = Trainer(image_loader, feature_extractor)
    solve_container = trainer.train(train_data, params.svm_params)

    test_data = list(first_class_test_data) + list(second_class_test_data)
    tester = Tester(image_loader, solve_container)
    return tester.test(test_data)


if __name__ == "__main__":
    print (run_customization(ImageLoader(image_dir_path=params.image_dir),
                             FeatureExtractor.load("color_feature_cache_5_10_6_6")))
    # print train_and_test(ImageLoader(image_dir_path=params.image_dir),
    # FeatureExtractor.load("color_feature_cache_5_10_6_6"))