import random
import sys
import math
import numba
import numpy.core.umath_tests as nt
import numpy as np
from numpy.linalg import norm
from numpy.lib.function_base import average
from numpy import amax

from .. feature_extractor import FeatureExtractor


class TextureFeatureExtractor(FeatureExtractor):
    ''' usecase:
    tfe = TextureFeatureExtractor()
    tfe.generate_tiles() # slow method
    features = tfe.extract(img) # very slow method
    7 second for extracting one feature
    '''
    def __init__(self, image_loader, tile_size=50, tiles_count=1,
                 images_for_tailing_count=500, delta=80.0, debug=False):
        self.il = image_loader
        self.tile_size = tile_size
        self.tiles_count = tiles_count
        self.images_for_tailing_count = images_for_tailing_count
        self.delta = delta
        self.debug = debug

    def extract(self, img):
        features = []
        for tile in self.tiles:
            features.append(self._img_tile_distance(img, tile))
        return features

    def extract_1(self, img):
        features = []
        for tile in self.tiles:
            features.append(self._img_tile_distance_1(img, tile))
        return features

    def extract_2(self, img):
        features = []
        for tile in self.tiles:
            features.append(self._img_tile_distance_2(img, tile))
        return features

    def extract_3(self, img):
        features = []
        for tile in self.tiles:
            features.append(self._img_tile_distance_3(img, tile))
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
        (img_height, img_width) = img.shape[0:2]
        (tile_height, tile_width) = tile.shape[0:2]
        result = sys.float_info.max
        for i in xrange(0, img_width - tile_width):
            for j in xrange(0, img_height - tile_height):
                sub_img = img[i:(i + tile_width), j:(j + tile_height)]
                dst = self._sub_img_tile_distance(tile, sub_img)
                result = min(dst, result)
        return result

    def _img_tile_distance_1(self, img, tile):
        (img_height, img_width) = img.shape[0:2]
        (tile_height, tile_width) = tile.shape[0:2]
        result = sys.float_info.max
        for i in xrange(0, img_width - tile_width):
            for j in xrange(0, img_height - tile_height):
                sub_img = img[i:(i + tile_width), j:(j + tile_height)]
                dst = self._debug_sub_img_tile_distance_1(tile, sub_img)
                result = min(dst, result)
        return result


    def _img_tile_distance_2(self, img, tile):
        (img_height, img_width) = img.shape[0:2]
        (tile_height, tile_width) = tile.shape[0:2]
        result = sys.float_info.max
        for i in xrange(0, img_width - tile_width):
            for j in xrange(0, img_height - tile_height):
                sub_img = img[i:(i + tile_width), j:(j + tile_height)]
                dst = self._debug_sub_img_tile_distance_2(tile, sub_img)
                result = min(dst, result)
        return result


    def _img_tile_distance_3(self, img, tile):
        (img_height, img_width) = img.shape[0:2]
        (tile_height, tile_width) = tile.shape[0:2]
        result = sys.float_info.max
        for i in xrange(0, img_width - tile_width):
            for j in xrange(0, img_height - tile_height):
                sub_img = img[i:(i + tile_width), j:(j + tile_height)]
                dst = self._debug_sub_img_tile_distance_3(tile, sub_img)
                result = min(dst, result)
        return result


    def _sub_img_tile_distance(self, tile, sub_img):
        return amax(norm(tile.astype(float) - sub_img.astype(float), axis=2).flat)

    def _debug_sub_img_tile_distance_1(self, tile, sub_img):
        distances = (tile.astype(float)-sub_img.astype(float))**2
        distances = distances.sum(axis=-1)
        distances = np.sqrt(distances)
        return amax(distances)

    def _debug_sub_img_tile_distance_2(self, tile, sub_img):
        delta = (tile.astype(float)-sub_img.astype(float))
        dist = np.sqrt(nt.inner1d(delta, delta))
        return amax(dist)

    def _debug_sub_img_tile_distance_3(self, tile, sub_img):
        delta = (tile.astype(float)-sub_img.astype(float))
        dist = np.sqrt(np.einsum('...ij, ...ij->...i', delta, delta))
        return amax(dist)

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
        print(images_for_tailing_names)
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

    def _debug_generate_tiles(self):
        '''
        generate tiles, and initialize self.tiles
        if images_for_tailing is too low raise exception
        :return:
        '''
        self.tiles = []
        available_images_names = self.il.available_images()
        images_for_tailing_names = ['dog.1067.jpg', 'dog.8873.jpg', 'cat.1673.jpg', 'dog.4086.jpg', 'cat.8664.jpg', 'dog.1516.jpg', 'cat.1801.jpg', 'cat.12494.jpg', 'dog.11074.jpg', 'dog.446.jpg', 'cat.6941.jpg', 'dog.10283.jpg', 'dog.9237.jpg', 'cat.1319.jpg', 'dog.1604.jpg', 'cat.8948.jpg', 'dog.5734.jpg', 'cat.5579.jpg', 'dog.8930.jpg', 'dog.912.jpg', 'dog.11287.jpg', 'dog.5571.jpg', 'dog.9044.jpg', 'dog.9349.jpg', 'cat.1538.jpg', 'cat.3359.jpg', 'dog.7438.jpg', 'dog.5346.jpg', 'dog.10604.jpg', 'dog.1171.jpg', 'cat.3026.jpg', 'dog.7491.jpg', 'dog.8296.jpg', 'cat.2139.jpg', 'cat.4733.jpg', 'cat.7477.jpg', 'dog.3870.jpg', 'dog.777.jpg', 'dog.8046.jpg', 'cat.4867.jpg', 'dog.8775.jpg', 'dog.664.jpg', 'dog.2258.jpg', 'cat.119.jpg', 'cat.12467.jpg', 'cat.11910.jpg', 'cat.461.jpg', 'dog.12038.jpg', 'dog.8798.jpg', 'dog.209.jpg', 'cat.2799.jpg', 'dog.12494.jpg', 'cat.2514.jpg', 'dog.2407.jpg', 'cat.10217.jpg', 'cat.8228.jpg', 'cat.6486.jpg', 'cat.8661.jpg', 'cat.10394.jpg', 'cat.4113.jpg', 'cat.11810.jpg', 'dog.10111.jpg', 'dog.6114.jpg', 'dog.6134.jpg', 'dog.2304.jpg', 'cat.10951.jpg', 'dog.9456.jpg', 'dog.5706.jpg', 'cat.2872.jpg', 'dog.5997.jpg', 'cat.136.jpg', 'dog.1730.jpg', 'cat.7481.jpg', 'dog.7847.jpg', 'dog.1191.jpg', 'cat.6069.jpg', 'cat.6116.jpg', 'cat.4282.jpg', 'cat.11243.jpg', 'dog.2827.jpg', 'cat.3313.jpg', 'dog.487.jpg', 'cat.4241.jpg', 'dog.149.jpg', 'dog.5551.jpg', 'dog.1552.jpg', 'cat.7478.jpg', 'dog.12325.jpg', 'cat.4780.jpg', 'dog.9367.jpg', 'dog.5328.jpg', 'cat.12331.jpg', 'dog.10707.jpg', 'cat.7564.jpg', 'cat.9037.jpg', 'cat.11903.jpg', 'dog.6706.jpg', 'dog.10426.jpg', 'cat.10085.jpg', 'cat.6635.jpg', 'cat.8983.jpg', 'cat.9972.jpg', 'dog.2614.jpg', 'cat.5377.jpg', 'cat.3627.jpg', 'cat.11814.jpg', 'cat.2020.jpg', 'dog.3061.jpg', 'dog.11631.jpg', 'cat.5970.jpg', 'dog.10631.jpg', 'cat.7075.jpg', 'dog.1115.jpg', 'dog.10885.jpg', 'dog.12071.jpg', 'cat.3005.jpg', 'dog.9745.jpg', 'dog.75.jpg', 'dog.6080.jpg', 'dog.12347.jpg', 'cat.5862.jpg', 'cat.7402.jpg', 'dog.1239.jpg', 'cat.5287.jpg', 'cat.11618.jpg', 'dog.9398.jpg', 'cat.568.jpg', 'cat.2536.jpg', 'cat.8255.jpg', 'cat.11460.jpg', 'cat.11109.jpg', 'dog.2012.jpg', 'dog.9654.jpg', 'dog.9917.jpg', 'cat.9047.jpg', 'dog.11976.jpg', 'cat.11004.jpg', 'dog.12474.jpg', 'cat.3263.jpg', 'dog.2315.jpg', 'dog.7976.jpg', 'dog.12072.jpg', 'cat.5253.jpg', 'dog.4639.jpg', 'cat.747.jpg', 'dog.890.jpg', 'cat.8688.jpg', 'dog.9787.jpg', 'cat.134.jpg', 'cat.9511.jpg', 'dog.10910.jpg', 'cat.542.jpg', 'cat.9586.jpg', 'dog.5323.jpg', 'dog.4644.jpg', 'dog.5693.jpg', 'dog.4329.jpg', 'cat.5416.jpg', 'cat.11763.jpg', 'cat.10356.jpg', 'dog.2136.jpg', 'dog.6252.jpg', 'cat.9593.jpg', 'dog.12032.jpg', 'cat.6861.jpg', 'cat.5033.jpg', 'cat.8051.jpg', 'dog.2325.jpg', 'cat.3115.jpg', 'cat.9924.jpg', 'dog.836.jpg', 'dog.203.jpg', 'cat.2346.jpg', 'cat.2737.jpg', 'cat.10536.jpg', 'cat.8956.jpg', 'dog.7613.jpg', 'dog.757.jpg', 'cat.7597.jpg', 'dog.8441.jpg', 'dog.2563.jpg', 'cat.11051.jpg', 'dog.317.jpg', 'dog.7862.jpg', 'cat.10810.jpg', 'cat.12376.jpg', 'cat.1142.jpg', 'cat.2350.jpg', 'cat.7167.jpg', 'dog.1371.jpg', 'cat.6159.jpg', 'dog.2783.jpg', 'cat.11019.jpg', 'cat.12093.jpg', 'cat.7491.jpg', 'cat.8407.jpg', 'cat.6151.jpg', 'cat.3358.jpg', 'cat.4135.jpg', 'dog.3641.jpg', 'cat.1371.jpg', 'dog.4443.jpg', 'dog.1822.jpg', 'cat.4824.jpg', 'dog.8129.jpg', 'cat.3070.jpg', 'cat.9946.jpg', 'cat.12249.jpg', 'dog.2098.jpg', 'cat.5565.jpg', 'cat.5389.jpg', 'dog.5155.jpg', 'cat.796.jpg', 'dog.10104.jpg', 'dog.3491.jpg', 'cat.9925.jpg', 'cat.3846.jpg', 'cat.7053.jpg', 'cat.6342.jpg', 'dog.2959.jpg', 'cat.372.jpg', 'dog.5404.jpg', 'dog.8189.jpg', 'dog.8266.jpg', 'dog.5074.jpg', 'dog.7867.jpg', 'dog.6133.jpg', 'dog.2779.jpg', 'dog.3457.jpg', 'cat.10715.jpg', 'cat.7159.jpg', 'cat.7844.jpg', 'cat.5839.jpg', 'cat.2245.jpg', 'dog.6730.jpg', 'dog.622.jpg', 'cat.3597.jpg', 'dog.6641.jpg', 'cat.6246.jpg', 'dog.2197.jpg', 'cat.8002.jpg', 'dog.384.jpg', 'dog.1361.jpg', 'cat.11314.jpg', 'dog.2349.jpg', 'cat.7263.jpg', 'dog.8137.jpg', 'dog.1954.jpg', 'dog.1505.jpg', 'cat.7805.jpg', 'dog.12422.jpg', 'dog.7437.jpg', 'cat.1655.jpg', 'cat.3072.jpg', 'dog.1054.jpg', 'dog.12116.jpg', 'cat.5438.jpg', 'dog.6116.jpg', 'dog.5582.jpg', 'dog.7595.jpg', 'dog.7319.jpg', 'cat.6900.jpg', 'cat.1162.jpg', 'dog.9881.jpg', 'cat.11924.jpg', 'cat.237.jpg', 'dog.2395.jpg', 'cat.8744.jpg', 'dog.11048.jpg', 'dog.9771.jpg', 'cat.8396.jpg', 'cat.2091.jpg', 'cat.947.jpg', 'cat.4197.jpg', 'dog.1943.jpg', 'dog.1742.jpg', 'dog.9653.jpg', 'cat.1675.jpg', 'dog.8350.jpg', 'cat.9010.jpg', 'dog.10958.jpg', 'dog.907.jpg', 'cat.4493.jpg', 'cat.1042.jpg', 'dog.10464.jpg', 'cat.5712.jpg', 'cat.6872.jpg', 'dog.11802.jpg', 'cat.3311.jpg', 'dog.6817.jpg', 'cat.4635.jpg', 'cat.8569.jpg', 'dog.11094.jpg', 'dog.3863.jpg', 'dog.3828.jpg', 'cat.10803.jpg', 'dog.11060.jpg', 'cat.4639.jpg', 'cat.5789.jpg', 'cat.1871.jpg', 'cat.1098.jpg', 'dog.4088.jpg', 'dog.3341.jpg', 'cat.3943.jpg', 'dog.12364.jpg', 'dog.2887.jpg', 'dog.10349.jpg', 'cat.5995.jpg', 'dog.10702.jpg', 'dog.6682.jpg', 'cat.5072.jpg', 'dog.11972.jpg', 'dog.4985.jpg', 'dog.3940.jpg', 'cat.3387.jpg', 'dog.8425.jpg', 'cat.8533.jpg', 'cat.7506.jpg', 'dog.5446.jpg', 'dog.9932.jpg', 'dog.5316.jpg', 'cat.6511.jpg', 'dog.8665.jpg', 'dog.11700.jpg', 'dog.12381.jpg', 'dog.10565.jpg', 'dog.2078.jpg', 'dog.6117.jpg', 'cat.10633.jpg', 'dog.8122.jpg', 'dog.391.jpg', 'dog.9180.jpg', 'dog.5306.jpg', 'dog.7858.jpg', 'cat.6981.jpg', 'dog.5051.jpg', 'cat.6735.jpg', 'cat.3389.jpg', 'dog.7436.jpg', 'dog.9334.jpg', 'dog.6694.jpg', 'dog.10736.jpg', 'dog.6245.jpg', 'dog.6393.jpg', 'cat.8600.jpg', 'cat.9636.jpg', 'cat.3301.jpg', 'dog.978.jpg', 'dog.1636.jpg', 'cat.9595.jpg', 'dog.138.jpg', 'dog.6647.jpg', 'dog.7238.jpg', 'cat.212.jpg', 'cat.165.jpg', 'cat.6014.jpg', 'cat.8154.jpg', 'dog.8808.jpg', 'dog.11179.jpg', 'dog.7478.jpg', 'cat.3050.jpg', 'cat.952.jpg', 'cat.139.jpg', 'dog.2440.jpg', 'dog.6507.jpg', 'cat.10594.jpg', 'cat.8613.jpg', 'dog.10178.jpg', 'cat.9656.jpg', 'dog.1602.jpg', 'cat.6654.jpg', 'cat.751.jpg', 'cat.6193.jpg', 'cat.2732.jpg', 'dog.5171.jpg', 'dog.8714.jpg', 'cat.9059.jpg', 'dog.5923.jpg', 'cat.3257.jpg', 'cat.10978.jpg', 'cat.12429.jpg', 'dog.12448.jpg', 'dog.1165.jpg', 'dog.4519.jpg', 'cat.7727.jpg', 'dog.3446.jpg', 'dog.8364.jpg', 'dog.3150.jpg', 'cat.7567.jpg', 'dog.6833.jpg', 'cat.3785.jpg', 'cat.5596.jpg', 'dog.10762.jpg', 'cat.9942.jpg', 'cat.9074.jpg', 'cat.5936.jpg', 'cat.1426.jpg', 'cat.2302.jpg', 'cat.1544.jpg', 'dog.4757.jpg', 'dog.6569.jpg', 'dog.9710.jpg', 'dog.2933.jpg', 'cat.3790.jpg', 'cat.12129.jpg', 'cat.4573.jpg', 'dog.5024.jpg', 'cat.4962.jpg', 'cat.2984.jpg', 'dog.2067.jpg', 'cat.10450.jpg', 'cat.5402.jpg', 'dog.10921.jpg', 'dog.7510.jpg', 'dog.10183.jpg', 'dog.4016.jpg', 'cat.4392.jpg', 'cat.12334.jpg', 'cat.7788.jpg', 'dog.918.jpg', 'cat.5722.jpg', 'dog.9538.jpg', 'dog.9277.jpg', 'cat.479.jpg', 'cat.11053.jpg', 'dog.10501.jpg', 'dog.8238.jpg', 'dog.841.jpg', 'dog.7304.jpg', 'dog.2161.jpg', 'dog.11984.jpg', 'dog.2808.jpg', 'cat.10469.jpg', 'dog.3298.jpg', 'dog.6723.jpg', 'dog.8419.jpg', 'dog.982.jpg', 'dog.6680.jpg', 'dog.6942.jpg', 'cat.1513.jpg', 'dog.12361.jpg', 'dog.3.jpg', 'cat.5692.jpg', 'cat.1712.jpg', 'cat.12055.jpg', 'dog.6592.jpg', 'cat.1992.jpg', 'dog.12321.jpg', 'cat.3275.jpg', 'dog.4746.jpg', 'cat.3540.jpg', 'cat.11355.jpg', 'cat.12316.jpg', 'dog.8920.jpg', 'cat.8515.jpg', 'cat.6746.jpg', 'dog.1310.jpg', 'dog.3849.jpg', 'dog.5361.jpg', 'cat.3118.jpg', 'dog.11461.jpg', 'dog.5622.jpg', 'cat.4542.jpg', 'dog.9976.jpg', 'dog.8836.jpg', 'cat.6663.jpg', 'cat.7580.jpg', 'dog.7335.jpg', 'dog.131.jpg', 'cat.6680.jpg', 'dog.6103.jpg', 'cat.4923.jpg', 'cat.4739.jpg', 'dog.1789.jpg', 'cat.11802.jpg', 'cat.8420.jpg', 'dog.11007.jpg', 'cat.9876.jpg', 'cat.3435.jpg', 'cat.4627.jpg', 'cat.11303.jpg', 'cat.9786.jpg', 'dog.8538.jpg', 'cat.7994.jpg', 'cat.1025.jpg', 'cat.1196.jpg', 'dog.7898.jpg', 'dog.11013.jpg', 'dog.8951.jpg', 'dog.11150.jpg', 'cat.11694.jpg', 'dog.10185.jpg', 'dog.10580.jpg', 'cat.10607.jpg', 'dog.9467.jpg', 'cat.9606.jpg', 'cat.7875.jpg', 'dog.5015.jpg', 'dog.5832.jpg', 'dog.306.jpg']

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