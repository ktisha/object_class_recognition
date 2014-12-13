__author__ = 'avesloguzova'

import logging

from svm_classifier import SVMClassifier
from solution_container import SolutionContainer
from tester import Tester


class Trainer(object):
    def __init__(self, image_loader, feature_extractor):
        self.image_loader = image_loader
        self.feature_extractor = feature_extractor

    def train(self, test_data, svm_params):
        """
            Start training for classifier
            :param
            images_names: names of images
            responses: expected responses for images
        """

        classifier = SVMClassifier(svm_params)
        solve_container = SolutionContainer(classifier, self.feature_extractor)
        test_features_vector = [self.feature_extractor.extract(self.image_loader.load(item[0])) for item in test_data]
        test_responses = [item[1] for item in test_data]
        logging.debug("Start training of classifier")
        classifier.train(test_features_vector, test_responses)

        return solve_container

    def svm_params_customization(self, data, default_svm_params, c_range, gamma_range):
        """

        :param data:
        :param default_svm_params:
        :param c_range:
        :param gamma_range:
        :return:
        """
        results = []
        for it, c in enumerate(c_range):
            if not it % 100:
                print(float(it) / len(list(c_range)))
            for gamma in gamma_range:
                default_svm_params["C"] = c
                default_svm_params["gamma"] = gamma
                logging.debug("Start cross validation with C = %e gamma = %e", c, gamma)
                results.append((c, gamma, self.k_fold_cross_validation(5, data, default_svm_params)))
        return results

    def k_fold_cross_validation(self, k, data, svm_params):
        """
        http://statweb.stanford.edu/~tibs/sta306b/cvwrong.pdf

        :param k:
        :param data:
        :param svm_params:
        :return:
        """

        partition = [1 for __ in range(0, k)]
        test_parts = Trainer.split_data(data, partition)
        quality = []

        for validate_part in test_parts:
            train_data = []
            for part in test_parts:
                if not part is validate_part:
                    for item in part:
                        train_data.append(item)
            solution = self.train(train_data, svm_params)
            quality.append(Tester(self.image_loader, solution).test(validate_part))
        logging.debug(quality)
        return sum(quality) / float(k)


    @staticmethod
    def split_data(sample, partition):
        """

        :param sample:
        :param partition:
        :return:
        """
        norm = sum(partition)
        last_index = 0
        result = []
        for part in partition:
            part_size = len(sample) * part / norm
            result.append(sample[last_index: last_index + part_size])
            last_index += part_size
        return tuple(result)
