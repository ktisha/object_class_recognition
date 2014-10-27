__author__ = 'avesloguzova'

import cv2


class SVMClassifier(object):
    def __init__(self, params):
        self.svm = cv2.SVM()
        self.params = params


    def train(self, train_data, responses):
        """
        Train SVM classifier
        :param train_data is array of features, responses is array of responses
        """
        self.svm.train(train_data, responses, self.params)

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

    def classify(self, test_data):
        """
        Classify object with SVM method
        :param test_data is numpy array of float32
        :return:
        """
        return self.svm.predict(test_data)