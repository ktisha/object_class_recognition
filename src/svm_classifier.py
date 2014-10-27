__author__ = 'avesloguzova'

import cv2

import numpy as np


class SVMClassifier(object):
    def __init__(self, params):
        self.svm = cv2.SVM()
        self.params = params

    def train(self, data, responses):
        """
        Train SVM classifier
        :param data is array of features
        :param responses is array of responses
        """
        training_data = np.float32(data)
        training_responses = np.float32(responses)
        self.svm.train(training_data, training_responses, params=self.params)

    def serialize(self, save_file):
        """
        save results of training to file
        :param save_file: path to file
        """

        self.svm.save(save_file)

    def deserialize(self, save_file):
        """
        load previous results of training from file
        :param save_file: path to file
        """
        self.svm.load(save_file)

    def classify(self, data):
        """
        Classify object with SVM method
        :param data is array of numpy array of float32
        :return:
        """
        test_data = np.float32(data)
        return self.svm.predict(test_data)