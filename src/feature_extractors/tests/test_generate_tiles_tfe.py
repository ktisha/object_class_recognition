from numpy.linalg import norm
from numpy.lib.function_base import average

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