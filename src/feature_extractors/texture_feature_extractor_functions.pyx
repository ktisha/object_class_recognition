import numpy as np
cimport numpy as np


def _image_tile_distance(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.uint8_t, ndim=3] tile):
    cdef size_t i, j
    cdef size_t x, y
    cdef float sq_result = float('inf')
    cdef float sq_dst
    cdef float sq_norm
    for i in range(img.shape[0] - tile.shape[0]):
        for j in range(img.shape[0] - tile.shape[1]):
            sq_dst = 0
            for x in range(tile.shape[0]):
                for y in range(tile.shape[1]):
                    sq_norm = (img[i + x, j + y, 0] - tile[x, y, 0]) * (img[i + x, j + y, 0] - tile[x, y, 0]) + \
                              (img[i + x, j + y, 1] - tile[x, y, 1]) * (img[i + x, j + y, 1] - tile[x, y, 1]) + \
                              (img[i + x, j + y, 2] - tile[x, y, 2]) * (img[i + x, j + y, 2] - tile[x, y, 2])
                    sq_dst = max(sq_dst, sq_norm)
            if sq_dst < sq_result:
                sq_result = sq_dst
    return pow(sq_result, 0.5)

