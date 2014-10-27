__author__ = 'Yurgin Pavel'
import cProfile
import cv2
from src.image_loader.image_loader import ImageLoader
import numpy as np
from src.feature_extractors.color_feature_extractor import ColorFeatureExtractor



DEBUG = 1
if DEBUG:
    color_extractor = ColorFeatureExtractor(parts_number=1, chanel_h=2, chanel_s=2, chanel_v=2)
    color_extractor._hsv = False
    image = np.ndarray(shape=(2, 2, 3), dtype=int)
    image[0, 0] = (0, 0, 0)
    image[0, 1] = (359, 255, 255)
    image[1, 0] = (180, 100, 100)
    image[1, 1] = (10, 100, 200)
    color_feature = color_extractor.extract(image)
    v = color_extractor._partition_of_color_space()

    if (color_feature != [1, 1, 0, 0, 1, 0, 0, 1]).all():
        assert "Wrong answer"

    image = np.ndarray(shape=(2, 2, 3), dtype=int)

    image[0, 0] = (179, 255, 127)
    image[0, 1] = (100, 255, 128)
    image[1, 0] = (359, 100, 200)
    image[1, 1] = (200, 220, 100)

    color_feature = color_extractor.extract(image)
    if (color_feature != [0, 0, 1, 1, 0, 1, 1, 0]).all():
        assert "Wrong answer"

    i_loader = ImageLoader(image_dir_path='../data/train')
    available_image = i_loader.available_images()
    image = i_loader.load(available_image[0])

    import random as r

    for i in xrange(10):
        parts = r.randint(1, len(image) // 25)
        while (len(image)) % parts:
            parts = r.randint(1, len(image) // 25)
        h = r.randint(1, 360 // 6)
        s = r.randint(1, 256 // 6)
        v = r.randint(1, 256 // 6)
        extractor = ColorFeatureExtractor(parts_number=parts, chanel_h=h, chanel_s=s, chanel_v=v)
        color_feature = extractor.extract(image)
        if len(color_feature) != h * s * v * parts ** 2:
            print(len(color_feature))
            print(h * s * v * parts ** 2)
            raise Exception("Error with h= {0} s= {1}, v= {2}, parts= {3} :".format(h, s, v, parts))
