import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euclidian_distance(double x1, double y1, double z1,
                               double x2, double y2, double z2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


@cython.boundscheck(False)
@cython.wraparound(False)
def _image_tile_distance(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.uint8_t, ndim=3] tile):
    """
    25 seconds by 1000 calls
    *distance between img and tile is defined as min of
    distances between tile and all sub-images of img
    *distance between a sub-image and a texture tile are defined
    as the maximum of Euclidian distance between their pixels in RGB
    :param img:
    :param tile:
    :return: distance in (0 ... 1)
    math.sqrt(255**2 + 255**2 + 255**2) = 441.67
    """
    cdef size_t i, j
    cdef size_t x, y
    cdef float result = float('inf')
    cdef float dst
    cdef float norm
    for i in range(img.shape[0] - tile.shape[0]):
        for j in range(img.shape[1] - tile.shape[1]):
            dst = 0
            for x in range(tile.shape[0]):
                for y in range(tile.shape[1]):
                    norm = euclidian_distance(img[i + x, j + y, 0], img[i + x, j + y, 1], img[i + x, j + y, 2],
                                              tile[x, y, 0], tile[x, y, 1], tile[x, y, 2])
                    dst = max(dst, norm)
            if dst < result:
                result = dst
    result /= sqrt(255.0**2 + 255.0**2 + 255.0**2)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _tile_tile_distance(np.ndarray[np.uint8_t, ndim=3] tile1, np.ndarray[np.uint8_t, ndim=3] tile2):
    """
    we define distance between two tiles as the average
    Euclidian distance between the pixels of the tiles in RGB(0..255, 0..255, 0..255)
    :param tile1:
    :param tile2:
    :return: distance in (0 ... 1)
    math.sqrt(255**2 + 255**2 + 255**2) = 441.67
    """
    cdef float result = 0.0
    cdef float dst = 0.0
    for i in range(tile1.shape[0]):
        for j in range(tile1.shape[1]):
            dst = euclidian_distance(tile1[i, j, 0], tile1[i, j, 1], tile1[i, j, 2],
                                     tile2[i, j, 0], tile2[i, j, 1], tile2[i, j, 2])
            result += dst
    result /= (tile1.shape[0] * tile1.shape[1])
    result /= sqrt(255.0**2 + 255.0**2 + 255.0**2)
    return result