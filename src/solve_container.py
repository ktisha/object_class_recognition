__author__ = 'avesloguzova'


class SolveContainer(object):
    def __init__(self, classifier, features_extractor):
        self.classifier = classifier
        self.features_extractor = features_extractor

    def serialize(self):
        """
        Serialize object
        :return:
        """