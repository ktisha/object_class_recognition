import abc
import zlib
import hashlib
import cPickle


class FeatureExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._cache = {}

    def extract(self, img):
        """
        :param img: 250*250 opencv image
        :return: feature vector
        """
        hash_code = hashlib.md5(img).hexdigest()
        feature_vector = self._get_cache(hash_code)

        if not len(feature_vector):
            feature_vector = self._extract(img)
            self._put_cache(hash_code, feature_vector)

        return feature_vector

    def save(self, path):
        """
        save feature extraction object to file
        :param path to file
        :return: None
        """
        with open(path, "w") as path:
            cPickle.dump(self, path, 2)

    @staticmethod
    def load(path):
        """
        load feature extraction object from file
        :param path: path to file
        :return: feature extraction object
        """
        with open(path, "r") as path:
            return cPickle.load(path)

    @abc.abstractmethod
    def _extract(self, image):
        return []

    def _get_cache(self, hash_code):
        """
        get cached feature vector
        :param hash_code: md5 code of image
        :return: np.array if cached object exist, [] otherwise
        """
        if hash_code in self._cache:
            return self._cache[hash_code]
        else:
            return []

    def _put_cache(self, hash_code, feature_vector):
        """
        put feature vector to _cache
        :param hash_code: md5 of image,
        :param feature_vector: feature vector
        :return: None
        """
        self._cache[hash_code] = feature_vector
