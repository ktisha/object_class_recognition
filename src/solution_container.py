__author__ = 'avesloguzova'


class SolutionContainer(object):
    def __init__(self, classifier, features_extractor):
        self.classifier = classifier
        self.features_extractor = features_extractor

    def save(self, filename_classifier, filename_extractor):
        """
        Serialize object
        :return:
        """
        self.features_extractor.save(filename_extractor)
        self.classifier.save(filename_classifier)

    def load(self, filename_classifier, filename_extractor):
        self.classifier.load(filename_classifier)
        self.features_extractor.load(filename_extractor)

    def process(self, img):
        return self.classifier.classify(self.features_extractor.extract(img))