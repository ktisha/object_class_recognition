__author__ = 'avesloguzova'
import svm_classifier


class TextureClassifier(svm_classifier):
    def __init__(self, params):
        """
        :param params:
        :return:
        """

    def train(self, map):
        """
        Train SVM classifier with color features
        :param map: dictionary of list of features to class
        """
