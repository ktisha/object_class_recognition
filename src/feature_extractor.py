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


class TextureFeatureExtractor(FeatureExtractor):
    def __init__(self):
        None
    def extract(self, img):
        return []


class ColorFeatureExtractor(FeatureExtractor):
    def __init__(self):
        None
    def extract(self, img):
        return []