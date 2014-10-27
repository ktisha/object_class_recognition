__author__ = 'avesloguzova'
from image_loader import image_loader as il
from feature_extractors import color_feature_extractor as cfexr

from svm_classifier import SVMClassifier
from solution_container import SolutionContainer
import params


def train(images, responses):
    """
        Start training for classifier
        :param
        images: names of images
        responses: expected responses for images
    """
    image_loader = il.ImageLoader(params.image_dir)
    # texture_feature_extractor = TextureFeatureExtractor(image_loader, params.tile_size, params.tiles_count,
    # params.images_for_tailing_count, params.delta, False)
    # texture_feature_extractor.generate_tiles()
    color_feature_extractor = cfexr.ColorFeatureExtractor()
    classifier = SVMClassifier(params.svm_params)
    solve_container = SolutionContainer(classifier, color_feature_extractor)
    test_data = [color_feature_extractor.extract(image_loader.load(img)) for img in images]
    classifier.train(test_data, responses)
    return solve_container










