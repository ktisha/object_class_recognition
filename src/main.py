import random
import logging

import params
from feature_extractors.feature_extractor import FeatureExtractor
from trainer import Trainer
from tester import Tester
from image_loader.image_loader import ImageLoader


logging.basicConfig(filename=params.logfile, level=logging.INFO)


def get_class_data(class_params, sample_size):
    """

    :param class_params:
    :param sample_size:
    :return:
    """
    indexes = random.sample(range(class_params["start_index"], class_params["image_count"]), sample_size)
    return map(
        lambda index: tuple([class_params["prefix"] + str(index) + class_params["postfix"], class_params["label"]]),
        indexes)


def get_train_and_test_data(class_params):
    """
    Method of generation names of test images for class
    :param class_params:
    :return: tuple of list of train and test data
    """
    sample = get_class_data(class_params, class_params["train_count"] + class_params["test_count"])
    return Trainer.split_data(sample, [class_params["train_count"], class_params["test_count"]])


def run_customization(image_loader, feature_extractor):
    logging.info("Start customize svm")
    logging.info("Generate sample")
    data = get_class_data(params.first_class_params, params.sample_size / 2) + get_class_data(
        params.second_class_params, params.sample_size / 2)
    random.shuffle(data)
    trainer = Trainer(image_loader, feature_extractor)
    c_range = [10 ** i for i in xrange(-5, 10)]
    gamma_range = [10 ** i for i in xrange(-5, 5)]
    results = trainer.svm_params_customization(data, params.svm_params, c_range, gamma_range)
    return results


def run_cross_validation(image_loader, feature_extractor):
    logging.info("Start 5-fold cross validation")
    logging.info("Generate sample")
    data = get_class_data(params.first_class_params, params.sample_size / 2) + get_class_data(
        params.second_class_params, params.sample_size / 2)
    random.shuffle(data)
    trainer = Trainer(image_loader, feature_extractor)
    return trainer.k_fold_cross_validation(5, data, params.svm_params, params.labels)



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
    result = run_cross_validation((ImageLoader(image_dir_path=params.image_dir),
                                   FeatureExtractor.load(params.features_cache)))
    # result = run_customization(ImageLoader(image_dir_path=params.image_dir),
    # FeatureExtractor.load(params.features_cache))
    # logging.info("RESULT:")
    # logging.info("  C   |   gamma   |   quality ")
    # [logging.info(" %s  |  %s | %s  ", c, gamma, q) for c, gamma, q in result]
    # print train_and_test(ImageLoader(image_dir_path=params.image_dir),
    # FeatureExtractor.load("color_feature_cache_5_10_6_6"))