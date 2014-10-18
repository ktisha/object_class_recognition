__author__ = 'avesloguzova'

import cv2


class SVMClassifier(object):
    def __init__(self, params, save_file):
        self.svm = cv2.SVM()
        self.params = params
        self.save_file = save_file

    def train(self, train_data, responses):
        """
        Train SVM classifier
        :param train_data is array of features, responses is array of responses
        """
        self.svm.train(train_data, responses, self.params)
        self.svm.save(self.save_file)


    def classify(self, test_data):
        """
        Classify object with SVM method
        :param features: test_data is list of features
        :return:
        """
        self.svm.load(self.save_file)
        return self.svm.svm.predict_all(test_data)