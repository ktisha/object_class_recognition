__author__ = 'Andrey Lazarevich'
import cv2
import numpy
import os


def nested_rect(candidate, bound):
    """
    Function to define nested contours.
    :param candidate: contours which we think can be nested in bound
    :param bound: bound around our candidate
    :return: True or False. depends on nested contours or not.
    """
    try:
        if candidate[0] <= bound[0] and candidate[1] <= bound[1]:
            if candidate[0] + candidate[2] <= bound[0] + bound[2] and candidate[1] + candidate[3] <= bound[1] + \
                    bound[3]:
                return True
            else:
                return False
        else:
            return False
    except IndexError:
        return False


def get_rects(gray):
    """
    Get two biggest bounding box of contours from grayscale image. Not used but can be helpful in future.
    :param gray: grayscale image
    :return: list of rectangles
    """
    rects = []
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_2 = 0
    max_contour_1 = 0
    for i in range(len(contours)):
        if len(contours[i]) > len(contours[max_contour_1]):
            max_contour_2 = max_contour_1
            max_contour_1 = i
    rect_1 = cv2.boundingRect(contours[max_contour_1])
    rect_2 = cv2.boundingRect(contours[max_contour_2])
    if nested_rect(rect_2, rect_1):
        rects.append(rect_2)
        return rects
    else:
        rects.append(rect_1)
        rects.append(rect_2)
        return rects


def get_conts(gray):
    """
    Grab two biggest contours from grayscale image
    :param gray: grayscale image
    :return: list of contours
    """
    conts = []
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_2 = 0
    max_contour_1 = 0
    for i in range(len(contours)):
        if len(contours[i]) > len(contours[max_contour_1]):
            max_contour_2 = max_contour_1
            max_contour_1 = i
    rect_1 = cv2.boundingRect(contours[max_contour_1])
    rect_2 = cv2.boundingRect(contours[max_contour_2])
    if nested_rect(rect_2, rect_1):
        conts.append(contours[max_contour_2])
        return conts
    else:
        conts.append(contours[max_contour_1])
        conts.append(contours[max_contour_2])
        return conts


class ImageLoader:
    def __init__(self, image_dir_path, cut_size=250, haarcascade_path='haarcascade_eye.xml'):
        """
        Constructor for image loader. Better to keep one or two instance of this class.
        :param image_dir_path: path to directory which contains some images to cut
        :param cut_size: height and weight of future cutted images
        :type cut_size: int
        :param haarcascade_path: path to haarcascade file ( optional )
        """
        self.haarcascade_name = haarcascade_path
        self.min_cont_len = 100
        self.cascade = []
        self.cut_size = cut_size
        self.image_ext = [".jpg", ".png", ".bmp"]
        self.image_dir_path = image_dir_path
        self.image_dir = self.__init_image_dir()
        self.cache_dir_path = os.path.join(image_dir_path, '../cache')
        self.cache_dir = []
        self.__init_cache_dir()

    def __init_haar(self):
        """

        Initialize haarcascade object. Not in use now, but who knows...
        """
        if self.haarcascade_name != '':
            self.cascade = cv2.CascadeClassifier(self.haarcascade_name)

    def load(self, image_name):
        """
        Load image from cached directory or cut it with one of the algorithms
        :param image_name: Name of image from available names set.
        :return: Numpy array, which contains cutted image
        """
        try:
            resized_image = self.__load_cached(image_name)
        except IOError:
            full_path = os.path.join(self.image_dir_path, image_name)
            image = cv2.imread(full_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            conts = get_conts(gray)
            x_local, y_local = self.__find_approx_center_in_conts_by_mass_center(conts)
            if x_local and y_local:
                cutted_image = self.__smart_cut(x_local, y_local, image)
            else:
                cutted_image = self.__simple_cut(image)
            resized_image = cv2.resize(cutted_image, (self.cut_size, self.cut_size))
            cv2.imwrite(os.path.join(self.cache_dir_path, image_name), resized_image)
            self.cache_dir.append(image_name)
        return resized_image

    def __load_cached(self, image_name):
        """
        Load image from already cached image in cache directory
        :param image_name: name of image
        :return: Cached image
        :raise IOError: Raised when cannot find this image in cache directory
        """
        if image_name in self.cache_dir:
            image = cv2.imread(os.path.join(self.cache_dir_path, image_name))
            return image
        else:
            raise IOError

    def __smart_cut(self, x_local, y_local, image):
        """
        Cut from local center point calculated from mass center.
        :param x_local: x coordinate of center
        :param y_local: y coordinate of center
        :param image: Image to cut from
        :return: cut_size x cut_size image.
        """
        height = image.shape[0]
        width = image.shape[1]
        if height < self.cut_size and width < self.cut_size:
            cutted_image = self.__simple_cut(image)
        else:
            if height < width:
                diff = height // 2
                right_space = width - x_local
                if right_space >= diff and x_local >= diff:
                    cutted_image = image[0:height, x_local - diff:x_local + diff]
                elif right_space < diff and x_local >= diff:
                    cutted_image = image[0:height, x_local - 2 * diff + right_space:width]
                elif right_space >= diff and x_local < diff:
                    cutted_image = image[0: height, 0: height]
                else:
                    cutted_image = self.__simple_cut(image)
            else:
                diff = width / 2
                top_space = height - y_local
                if top_space >= diff and y_local >= diff:
                    cutted_image = image[y_local - diff: y_local + diff, 0:width]
                elif top_space < diff and y_local >= diff:
                    cutted_image = image[y_local - 2 * diff + top_space: height, 0:width]
                elif top_space >= diff and y_local < diff:
                    cutted_image = image[0:width, 0:width]
                else:
                    cutted_image = self.__simple_cut(image)
        return cutted_image

    def __simple_cut(self, image):
        """
        Cut image from the center. No magic
        :param image: Image to cut
        :return: cut_size x cut_size image
        """
        height = image.shape[0]
        width = image.shape[1]
        x_local = width // 2
        y_local = height // 2
        if height < width:
            diff = height // 2
            cutted_image = image[0:height, x_local - diff:x_local + diff]
        else:
            diff = width // 2
            cutted_image = image[y_local - diff:y_local + diff, 0:width]
        return cutted_image

    def available_images(self):
        """

        Get list of available image names
        :return: list with available names
        """
        return self.image_dir

    def __init_cache_dir(self):
        """

        Initialize cached directory. If it doesn't exist - creates it.
        """
        if not os.path.exists(self.cache_dir_path):
            os.mkdir(self.cache_dir_path)
        else:
            self.cache_dir = os.listdir(self.cache_dir_path)

    def __init_image_dir(self):
        """

        Initialize directory with image set.
        :return: List of available image names.
        """
        image_names = []
        all_files = os.listdir(self.image_dir_path)
        for some_file in all_files:
            filext = os.path.splitext(some_file)[1]
            if filext in self.image_ext:
                image_names.append(some_file)
        return image_names

    def __find_approx_center_in_conts_by_mass_center(self, conts):
        """
        Find mass center of contours and get mean value
        :param conts: contours to find mass center.
        :return: x and y coordinates of mean value of mass centers
        """
        center_x = 0
        center_y = 0
        centers = []
        for cont in conts:
            if cont.size > self.min_cont_len:
                m = cv2.moments(cont)
                cx = int(m['m10']/m['m00'])
                cy = int(m['m01']/m['m00'])
                centers.append((cx, cy))
        for center in centers:
            center_x += center[0]
            center_y += center[1]
        if centers:
            center_x /= len(centers)
            center_y /= len(centers)
        return center_x, center_y