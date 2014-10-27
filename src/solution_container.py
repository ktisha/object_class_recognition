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
        self.features_extractor.serialize(filename)
        self.classifier.serialize(filename)

    def deserialize(self, filename):
        pass

    def process(self, img):
        return self.classifier.classify(self.features_extractor.extract(img))