__author__ = 'avesloguzova'


class Tester(object):
    @staticmethod
    def default_quality(actual, excepted):
        return float(sum(map(lambda x: x[0] == x[1], zip(actual, excepted)))) / len(actual)

    def __init__(self, image_loader, solution_container, quality_function=default_quality):
        self.image_loader = image_loader
        self.solution_container = solution_container
        self.quality = quality_function

    def test(self, test_data):
        """
        Test result of training
        :param test_data:
        :return:
        """

        results = [self.test_image(item) for item, __ in test_data]
        correct = Tester.default_quality(results, [excepted_response for __, excepted_response in test_data])
        return correct

    def test_image(self, image_name):
        """
        Classify single image
        :param image_name: name of image in loader directory
        :return: label of class for image
        """
        print("Test " + image_name)
        img = self.image_loader.load(image_name)
        res = self.solution_container.process(img)
        print("result:" + str(res))
        return res