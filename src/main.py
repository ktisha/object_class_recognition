import random
import logging

import params
from feature_extractors.feature_extractor import FeatureExtractor
from trainer import Trainer
from tester import Tester
from image_loader.image_loader import ImageLoader
from image_loader.dataset_loader import DatasetLoader


logging.basicConfig(filename=params.logfile, level=logging.INFO)


def get_class_data(data_loader, class_params):
    """


    :param data_loader:
    :param class_params:
    :return:
    """
    # indexes = random.sample(range(class_params["start_index"], class_params["image_count"]), sample_size)
    # return map(
    # lambda index: tuple([class_params["prefix"] + str(index) + class_params["postfix"], class_params["label"]]),
    # indexes)
    data = data_loader.get_dataset(class_params["name"], class_params["sample_size"])
    labels = [class_params["label"] for __ in range(class_params["sample_size"])]
    return zip(data, labels)


def get_train_and_test_data(data_loader, class_params):
    """
    Method of generation names of test images for class
    :param class_params:
    :return: tuple of list of train and test data
    """
    sample = get_class_data(data_loader, class_params)
    return Trainer.split_data(sample, [80, 20])


def run_customization(dataset_loader, feature_extractor):
    logging.info("Start customize svm")
    logging.info("Generate sample")
    data = []
    for class_params in params.class_params:
        data += get_class_data(dataset_loader, class_params)
    random.shuffle(data)
    image_loader = ImageLoader(image_dir_path=dataset_loader.dataset_dir)
    trainer = Trainer(image_loader, feature_extractor)
    c_range = [10 ** i for i in xrange(-5, 10)]
    gamma_range = [10 ** i for i in xrange(-5, 5)]
    results = trainer.svm_params_customization(data, params.svm_params, c_range, gamma_range)
    return results


def train_and_test(dataset_loader, feature_extractor):
    """
    Simple implementation of train and test function
    :param feature_extractor:
    """
    train_data = []
    test_data = []
    for class_params in params.class_params:
        class_train_data, class_test_data = get_train_and_test_data(dataset_loader, class_params)
        train_data += class_train_data
        test_data += class_test_data
    image_loader = ImageLoader(dataset_loader.dataset_dir)
    trainer = Trainer(image_loader, feature_extractor)
    solve_container = trainer.train(train_data, params.svm_params)
    tester = Tester(image_loader, solve_container)
    return tester.test(test_data)


if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    result = run_customization(dataset_loader,
                               FeatureExtractor.load(params.features_cache))
    logging.info("RESULT:")
    logging.info("  C   |   gamma   |   quality ")
    [logging.info(" %s  |  %s | %s  ", c, gamma, q) for c, gamma, q in result]
    # print train_and_test(data_loader, FeatureExtractor.load(params.features_cache))