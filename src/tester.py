__author__ = 'avesloguzova'

import logging


class Tester(object):
    @staticmethod
    def default_quality(actual, excepted):
        return float(sum(map(lambda x: x[0] == x[1], zip(actual, excepted)))) / len(actual)

    @staticmethod
    def f1_score(actual, excepted, label):
        true_positives = 0.0
        false_negatives = 0.0
        false_positives = 0.0
        for act, exc in zip(actual, excepted):
            if exc == label:
                if act == exc:
                    true_positives += 1
                else:
                    false_negatives += 1
            elif act == exc:
                false_positives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return 2.0 * precision * recall / (precision + recall)

    def __init__(self, image_loader, solution_container, quality_function=default_quality):
        self.image_loader = image_loader
        self.solution_container = solution_container
        self.quality = quality_function

    def test(self, test_data, labels=None):
        """
        Test result of training
        :param test_data:
        :return:
        """

        results = [self.test_image(item) for item, __ in test_data]
        excepted = [excepted_response for __, excepted_response in test_data]
        if labels:
            for label in labels:
                logging.info(Tester.f1_score(results, excepted, label))
        correct = self.quality(results, excepted)
        return correct

    def test_image(self, image_name):
        """
        Classify single image
        :param image_name: name of image in loader directory
        :return: label of class for image
        """
        logging.debug("Test " + image_name)
        img = self.image_loader.load(image_name)
        res = self.solution_container.process(img)
        logging.debug("result:" + str(res))
        return res