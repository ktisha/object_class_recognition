import cv2
import random
import sys

from numpy.linalg import norm
from numpy.lib.function_base import average
from numpy import amax

from feature_extractor import FeatureExtractor
from texture_feature_extractor_functions import _image_tile_distance


class TextureFeatureExtractor(FeatureExtractor):
    ''' usecase:
    tfe = TextureFeatureExtractor()
    tfe.generate_tiles() # slow method
    features = tfe.extract(img) # very slow method
    7 second for extracting one feature
    '''
    def __init__(self, image_loader, tile_size=5, tiles_count=1000,
                 images_for_tailing_count=500, delta=0.0, debug=False):
        self.il = image_loader
        self.tile_size = tile_size
        self.tiles_count = tiles_count
        self.images_for_tailing_count = images_for_tailing_count
        self.delta = delta
        self.debug = debug

    def extract(self, img):
        features =[]
        for tile in self.tiles:
            # print('python: {}'.format(self._img_tile_distance(img, tile)))
            # print('cython: {}'.format(_image_tile_distance(img, tile)))
            # cv2.imshow('test_tile_tile_distance', tile)
            # cv2.waitKey()
            features.append(_image_tile_distance(img, tile))
        return features

    def _img_tile_distance(self, img, tile):
        '''
        distance between img and tile is defined as min of
        distances between tile and all sub-images of img
        :param img:
        :param tile:
        :return: distance in (0 ... sqrt(255**2 + 255**2 + 255**2))
        math.sqrt(255**2 + 255**2 + 255**2) = 441.67
        '''
        img_height, img_width = img.shape[0:2]
        tile_height, tile_width = tile.shape[0:2]
        result = sys.float_info.max
        for i in xrange(0, img_width - tile_width):
            for j in xrange(0, img_height - tile_height):
                sub_img = img[i:(i + tile_width), j:(j + tile_height)]
                dst = self._sub_img_tile_distance(tile, sub_img)
                result = min(dst, result)
        return result

    def _sub_img_tile_distance(self, tile, sub_img):
        '''
        distance between a sub-image and a texture tile are defined
        as the maximum of Euclidian distance between their pixels in RGB
        :param tile:
        :param sub_img:
        :return: distance
        '''
        return amax(norm(tile.astype(float) - sub_img.astype(float), axis=2).flat)

    def _tile_tile_distance(self, tile1, tile2):
        '''
        we define distance between two tiles as the average
         Euclidian distance between the pixels of the tiles in RGB(0..255, 0..255, 0..255)
        :param tile1:
        :param tile2:
        :return: distance in (0 ... sqrt(255**2 + 255**2 + 255**2))
        math.sqrt(255**2 + 255**2 + 255**2) = 441.67
        '''
        return average(norm(tile1.astype(float) - tile2.astype(float), axis=2))

    def _check_same_tile_exist(self, new_tile):
        for tile in self.tiles:
            dst = self._tile_tile_distance(tile, new_tile)
            self._debug_print('dst: {}'.format(dst))
            if dst < self.delta:
                return True
        return False

    def generate_tiles(self):
        '''
        generate tiles, and initialize self.tiles
        if images_for_tailing is too low raise exception
        :return:
        '''
        self.tiles = []
        available_images_names = self.il.available_images()
        images_for_tailing_names = random.sample(available_images_names, self.images_for_tailing_count)
        for img_name in images_for_tailing_names:
            img = self.il.load(img_name)
            new_tiles = self._divide_img_by_tiles(img)
            for new_tile in new_tiles:
                if not self._check_same_tile_exist(new_tile):
                    self.tiles.append(new_tile)
                    self._debug_print('tiles count: {}'.format(len(self.tiles)))
                    if len(self.tiles) == self.tiles_count:
                        return
        raise Exception('Tiles not generated')

    def _divide_img_by_tiles(self, img):
        tiles = []
        (height, width) = img.shape[0:2]
        for x in xrange(0, width, self.tile_size):
            for y in xrange(0, height, self.tile_size):
                tiles.append(img[x:(x + self.tile_size), y:(y + self.tile_size)])
        return tiles

    def _debug_print(self, str):
        if self.debug:
            print('TFE debug: {}'.format(str))