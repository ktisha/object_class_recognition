__author__ = 'avesloguzova'
from image_loader import image_loader as im
import params


def test(images, responses, solution_container):
    """
    Test result of training
    :param images: list of images name
    :param responses: list of responses
    :param solution_container: object of SolutionContainer from solution_container
    :return:
    """
    image_loader = im.ImageLoader(params.image_dir)
    image_loader.load(images[0])
    imgs = [image_loader.load(img) for img in images]
    results = [solution_container.process(img) for img in imgs]
    correct = sum(map(lambda x: x[0] == x[1], zip(results, responses))) * 100 / len(responses)
    print correct
