import numpy as np

from src.image_loader.image_loader import ImageLoader
from src.feature_extractors.texture_feature_extractor import TextureFeatureExtractor

import cProfile


import sys
from numpy.linalg import norm
from numpy import amax


def _img_tile_distance(self, img, tile):
    """
    distance between img and tile is defined as min of
    distances between tile and all sub-images of img
    :param img:
    :param tile:
    :return: distance in (0 ... sqrt(255**2 + 255**2 + 255**2))
    math.sqrt(255**2 + 255**2 + 255**2) = 441.67
    """
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
    """
    distance between a sub-image and a texture tile are defined
    as the maximum of Euclidian distance between their pixels in RGB
    :param tile:
    :param sub_img:
    :return: distance
    """
    return amax(norm(tile.astype(float) - sub_img.astype(float), axis=2).flat)


def py_extract(self, img):
    features = np.empty((self.tiles_count,), dtype=float)
    for index, tile in enumerate(self.tiles):
        features[index] = self._img_tile_distance(img, tile)
    return features


def test():
    il = ImageLoader(image_dir_path='../../../data/train')
    TextureFeatureExtractor._img_tile_distance = _img_tile_distance
    TextureFeatureExtractor._sub_img_tile_distance = _sub_img_tile_distance
    TextureFeatureExtractor.py_extract = py_extract
    tfe = TextureFeatureExtractor(il, tiles_count=3, delta=20.0)
    tfe.generate_tiles()
    img = il.load(il.available_images()[100])
    print('python: {}'.format(tfe.py_extract(img)))
    print('cython: {}'.format(tfe.extract(img)))


def main():
    use_cprofile = True
    if use_cprofile:
        cProfile.run('test()')
    else:
        test()


if __name__ == "__main__":
    main()