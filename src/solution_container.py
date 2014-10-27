__author__ = 'avesloguzova'


class SolutionContainer(object):
    def __init__(self, classifier, features_extractor):
        self.classifier = classifier
        self.features_extractor = features_extractor

    def serialize(self, filename):
        """
        Serialize object
        :return:
        """
        self.features_extractor.serialize()
        self.classifier.serialize()

    def deserialize(self, filename):
        self.features_extractor.deserialize()
        self.classifier.deserialize()

    def get_classifier(self):
        return self.classifier

    def get_extractor(self):
        return self.features_extractor