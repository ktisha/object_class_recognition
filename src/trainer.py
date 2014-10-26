__author__ = 'avesloguzova'
from image_loader import ImageLoader
from texture_feature_extractor import TextureFeatureExtractor
from svm_classifier import SVMClassifier
from solve_container import SolveContainer
import params


def train(images, responses):
    """
        Start trainig for classifier
        :param
        images: names of images
        responses: expected responses for images
        """
    image_loader = ImageLoader(params.image_dir)

    texture_feature_extractor = TextureFeatureExtractor(image_loader)
    texture_feature_extractor.generate_tiles()

    classifier = SVMClassifier(params.svm_params, params.svm_save_file)

    solve_container = SolveContainer(classifier, texture_feature_extractor)
    test_data = [texture_feature_extractor.extract(img) for img in images]
    classifier.train(test_data, responses)










