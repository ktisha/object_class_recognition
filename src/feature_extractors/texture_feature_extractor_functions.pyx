import numpy as np
cimport numpy as np


def _image_tile_distance(np.ndarray[np.float64_t, ndim=3] img, np.ndarray[np.float64_t, ndim=3] tile):
    cdef size_t i, j
    cdef size_t x, y
    cdef float result = float('inf')
    cdef float dst
    cdef float norm
    for i in range(img.shape[0] - tile.shape[0]):
        for j in range(img.shape[0] - tile.shape[1]):
            dst = 0
            for x in range(tile.shape[0]):
                for y in range(tile.shape[1]):
                    sq_norm = (img[i + x, j + y, 0] - tile[x, y, 0]) * (img[i + x, j + y, 0] - tile[x, y, 0]) + \
                                (img[i + x, j + y, 1] - tile[x, y, 1]) * (img[i + x, j + y, 1] - tile[x, y, 1]) + \
                                (img[i + x, j + y, 2] - tile[x, y, 2]) * (img[i + x, j + y, 2] - tile[x, y, 2])
                    dst = max(dst, pow(sq_norm, 0.5))
            if dst < result:
                result = dst
    return result

