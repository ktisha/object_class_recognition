import sys


from feature_extractor import FeatureExtractor


class TextureFeatureExtractor(FeatureExtractor):
    def __init__(self, image_loader):
        self.image_loader = image_loader

