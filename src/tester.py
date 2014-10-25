__author__ = 'avesloguzova'
from image_loader import ImageLoader
import params


def test(images, responses, solve_container):
    image_loader = ImageLoader(params.image_dir)
    process = lambda x: \
        solve_container.get_classifier.classify(solve_container.get_extractor().extract(image_loader.load(x)))
    # maybe replace this long long lambda? By this for example.
    # def process(image):
    #     extracted = solve_container.get_extractor().extract(image_loader.load(x))
    #     return solve_container.get_classifier.classify(extracted)

    results = map(process, images)
    correct = sum(map(lambda x: x[0] == x[1], zip(results, responses))) * 100 / len(responses) # or 100.0 for float
    # or
    # correct = sum(for i, j in zip(results, responses) if i == j) * 100. / len(responses)
    # remember, that zip return list in python2 and return like-iterator object in python3
    # use itertools.izip if you want to return like-iterator object
    print correct
