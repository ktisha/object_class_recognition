import random

from numpy.linalg import norm

from feature_extractor import FeatureExtractor


class TextureFeatureExtractor(FeatureExtractor):
    def __init__(self, image_loader):
        self.il = image_loader
        self.tile_size = 50
        self.tiles_count = 100
        self.images_for_tailing_count = 500
        self.delta = 40.0
        self._generate_tiles()

    def extract(self, img):
        return []

    def _tile_dist(self, tile1, tile2):
        sum_dist = 0
        diff = tile1 - tile2
        for row in diff:
            for pixel in row:
                pixel /= 255
                sum_dist += norm(pixel)
        return sum_dist / (self.tile_size ** 2)

    def _check_same_tile_exist(self, new_tile):
        for tile in self.tiles:
            if self._tile_dist(tile, new_tile) < self.delta:
                return True
        return False

    def _generate_tiles(self):
        self.tiles = []
        available_images_names = self.il.available_images()
        images_for_tailing_names = random.sample(available_images_names, self.images_for_tailing_count)
        for img_name in images_for_tailing_names:
            img = self.il.load(img_name)
            new_tiles = self._divide_img_by_tiles(img)
            for new_tile in new_tiles:
                if not self._check_same_tile_exist(new_tile):
                    self.tiles.append(new_tile)
                    if len(self.tiles) == self.tiles_count:
                        return

    def _divide_img_by_tiles(self, img):
        tiles = []
        height = img.shape[0]
        width = img.shape[1]
        for x in xrange(0, width, self.tile_size):
            for y in xrange(0, height, self.tile_size):
                tiles.append(img[x:(x + self.tile_size), y:(y + self.tile_size)])
        return tiles
