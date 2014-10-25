__author__ = 'avesloguzova'
from image_loader import ImageLoader
import params


def test(images, responses, solve_container):
    image_loader = ImageLoader(params.image_dir)
    process = lambda x: \
        solve_container.get_classifier.classify(solve_container.get_extractor().extract(image_loader.load(x)))
    results = map(process, images)
    correct = sum(map(lambda x: x[0] == x[1], zip(results, responses))) * 100 / len(responses)
    print correct
