import time


from src.image_loader.image_loader import ImageLoader
import src.feature_extractors.texture_feature_extractor as TFE

import cProfile

from numpy.linalg import norm
from numpy.lib.function_base import average


def _tile_tile_distance(self, tile1, tile2):
    """
    we define distance between two tiles as the average
     Euclidian distance between the pixels of the tiles in RGB(0..255, 0..255, 0..255)
    :param tile1:
    :param tile2:
    :return: distance in (0 ... sqrt(255**2 + 255**2 + 255**2))
    math.sqrt(255**2 + 255**2 + 255**2) = 441.67
    """
    return average(norm(tile1.astype(float) - tile2.astype(float), axis=2))


def test():
    il = ImageLoader(image_dir_path='../../../data/train')
    tfe = TFE.TextureFeatureExtractor(il, tiles_count=1000, delta=80.0, debug=True)
    start = time.time()
    tfe.generate_tiles()
    stop = time.time()
    tfe.show_tiles()
    print('tile generation: {}'.format(stop - start))


def main():
    use_cprofile = False
    if use_cprofile:
        cProfile.run('test()')
    else:
        test()


if __name__ == "__main__":
    main()