import cv2
from feature_extractor import FeatureExtractor
import numpy as np
import warnings
from itertools import izip

class ColorFeatureExtractor(FeatureExtractor):
    def __init__(self, n=10, chanel_h=10, chanel_s=10, chanel_v=10):
        self._chanel_h = chanel_h
        self._chanel_s = chanel_s
        self._chanel_v = chanel_v
        self.n = n

    def extract(self, image):
        """
        Extract color feature vector from image
        :param image: cv2 image
        :return: vector of bool with length = n * count of h chanel parts * count of s chanel parts *
                 * count v chanel parts, where n is count of image parts.
        """
        if image.shape[0] < self.n or image.shape[1] < self.n:
            raise Exception("size of image must be bigger then n")
        if len(image) % self.n:
            warnings.warn("Warning: n is not multiple of size of image")
        feature_vector = []
        for part_of_image in self._partition_image_generator(image):
            feature_vector.extend(self._get_feature_vector(part_of_image).tolist())
        return feature_vector


    def _partition_image_generator(self, image):
        """
        :param image: square cv2 image
        :return: part of image, converted to HSV color space
        """
        #hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        hsv_image = image
        image_size = len(image)
        part_size = image_size // self.n
        for up_left, up_right in zip(xrange(0, image_size, part_size),
                                     xrange(part_size, image_size + part_size, part_size)):
            for down_left, down_right in zip(xrange(0, image_size, part_size),
                                             xrange(part_size, image_size + part_size, part_size)):
                yield hsv_image[up_left:up_right, down_left:down_right]

    def _get_feature_vector(self, image):
        """
        :param image: image
        :return: feature array of bool
        """
        pixel_feature = np.zeros((self._chanel_h, self._chanel_s, self._chanel_v), np.bool)

        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                h, s, v = image[i, j]
                h_chanel_coordinate = h // round(360. / self._chanel_h)
                s_chanel_coordinate = s // round(256. / self._chanel_s)
                v_chanel_coordinate = v // round(256. / self._chanel_v)
                pixel_feature[h_chanel_coordinate, s_chanel_coordinate, v_chanel_coordinate] = 1
        return pixel_feature.reshape(self._chanel_h * self._chanel_s * self._chanel_v)

    def _partirion_of_color_space(self):
        _chanel_h_range = 360
        _chanel_s_range = 256
        _chanel_v_range = 256
        _chanel_h_step = _chanel_h_range // self._chanel_h
        _chanel_s_step = _chanel_s_range // self._chanel_s
        _chanel_v_step = _chanel_v_range // self._chanel_v
        _chanel_h_partition = [i for i in xrange(0 + _chanel_h_step, _chanel_h_range + _chanel_h_step, _chanel_h_step)]
        _chanel_s_partition = [i for i in xrange(0 + _chanel_s_step, _chanel_s_range + _chanel_s_step, _chanel_s_step)]
        _chanel_v_partition = [i for i in xrange(0 + _chanel_v_step, _chanel_v_range + _chanel_v_step, _chanel_s_step)]
        return [_chanel_h_partition, _chanel_s_partition, _chanel_v_partition]

