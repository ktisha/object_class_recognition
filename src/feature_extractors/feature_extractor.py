import abc


class FeatureExtractor(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def extract(self, img):
        """
        :param img: 250*250 opencv image
        :return: feature vector
        """
        return