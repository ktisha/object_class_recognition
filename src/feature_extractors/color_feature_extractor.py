import cv2
from feature_extractor import FeatureExtractor
import numpy as np
import warnings
from itertools import izip
from math import ceil

class ColorFeatureExtractor(FeatureExtractor):
    def __init__(self, parts_number=1, chanel_h=10, chanel_s=10, chanel_v=10, ):
        """
        :param parts_number: image will be divided into (parts_number**2) parts.
        Number of parts must be less then number of pixel.
        Recommended that the variable was a multiple linear image size.
        :param chanel_h: number of hue chanel parts
        :param chanel_s: number of saturation chanel parts
        :param chanel_v: number of value chanel parts
        """
        super(ColorFeatureExtractor, self).__init__()
        self._chanel_h = chanel_h
        self._chanel_s = chanel_s
        self._chanel_v = chanel_v
        self._parts_number = parts_number
        self._h_range = 360
        self._s_range = 256
        self._v_range = 256
        self._hsv = True

    def _extract(self, image):
        """
        Extract color feature vector from image
        :param image: square cv2 image
        :return: vector of bool with length = (parts of image**2) * count of h chanel parts * count of s chanel parts *
                 * count v chanel parts
        """
        if image.shape[0] < self._parts_number or image.shape[1] < self._parts_number:
            raise IndexError("Size of image must be bigger then parts_number")
        if len(image) % self._parts_number:
            warnings.warn("Warning: parts_number is not multiple of linear image size")

        subvector_length = self._chanel_h * self._chanel_s * self._chanel_v
        length_of_result = subvector_length * self._parts_number**2
        feature_vector = np.empty((length_of_result,), dtype=bool)
        indexes = (i for i in xrange(0, length_of_result, subvector_length))

        for idx, part_of_image in izip(indexes, self._partition_image_generator(image)):
            subvector = self._get_feature_vector(part_of_image)
            for i, value in enumerate(subvector):
                feature_vector[idx + i] = value
        return feature_vector

    def _partition_image_generator(self, image):
        """
        :param image: square(!) cv2 image
        :return: part of image, converted to HSV color space
        """
        if self._hsv:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        else:
            hsv_image = image  # for debug

        image_size = len(image)
        part_size = image_size // self._parts_number
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
        h_range = np.int(ceil(self._h_range / float(self._chanel_h)))
        s_range = np.int(ceil(self._s_range / float(self._chanel_s)))
        v_range = np.int(ceil(self._v_range / float(self._chanel_v)))
        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                h, s, v = image[i, j]
                h_chanel_coordinate = np.int(h) // h_range
                s_chanel_coordinate = np.int(s) // s_range
                v_chanel_coordinate = np.int(v) // v_range
                pixel_feature[h_chanel_coordinate, s_chanel_coordinate, v_chanel_coordinate] = True
        return pixel_feature.reshape(self._chanel_h * self._chanel_s * self._chanel_v)

    def _partition_of_color_space(self):
        _chanel_h_step = self._h_range // self._chanel_h
        _chanel_s_step = self._s_range // self._chanel_s
        _chanel_v_step = self._v_range // self._chanel_v
        _chanel_h_partition = [i for i in xrange(0 + _chanel_h_step, self._h_range + _chanel_h_step, _chanel_h_step)]
        _chanel_s_partition = [i for i in xrange(0 + _chanel_s_step, self._s_range + _chanel_s_step, _chanel_s_step)]
        _chanel_v_partition = [i for i in xrange(0 + _chanel_v_step, self._v_range + _chanel_v_step, _chanel_s_step)]
        return [_chanel_h_partition, _chanel_s_partition, _chanel_v_partition]

