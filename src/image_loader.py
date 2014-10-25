__author__ = 'Andrey Lazarevich'
import cv2
import os


class ImageLoader:

    def __init__(self, image_dir_path=os.getcwd(), cut_size=250, haarcascade_path='haarcascade_eye.xml'):
        self.separator = os.sep
        self.haarcascade_name = haarcascade_path
        self.cascade = []
        self.__init_haar()
        assert isinstance(cut_size, int)
        self.cut_size = cut_size
        self.image_ext = [".jpg", ".png", ".bmp"]
        self.image_dir_path = image_dir_path
        self.image_dir = self.__init_image_dir()
        self.cache_dir_path = os.path.join(os.getcwd(), 'cache')
        self.cache_dir = []
        self.__init_cache_dir()

    def __init_haar(self):
        try:
            self.cascade = cv2.CascadeClassifier(self.haarcascade_name)
        except Exception:
            self.cascade = []

    def load(self, image_name):
        try:
            resized_image = self.__load_cached(image_name)
        except IndexError:
            full_path = self.image_dir_path + self.separator + image_name
            image = cv2.imread(full_path)
            if not self.cascade:
                cutted_image = self.__simple_cut(image)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                eyes = self.cascade.detectMultiScale(gray)
                try:
                    x_local = eyes[0][0]
                    y_local = eyes[0][1]
                    cutted_image = self.__smart_cut(x_local, y_local, image)
                except IndexError:
                    cutted_image = self.__simple_cut(image)
            resized_image = cv2.resize(cutted_image, (self.cut_size, self.cut_size))
            cv2.imwrite(self.cache_dir_path + self.separator + image_name, resized_image)
            self.cache_dir.append(image_name)
        finally:
            return resized_image

    def __load_cached(self, image_name):
        if image_name in self.cache_dir:
            image = cv2.imread(self.cache_dir_path + self.separator + image_name)
            return image
        else:
            raise IndexError

    def __smart_cut(self, x_local, y_local, image):
        height = image.shape[0]
        width = image.shape[1]
        if height < width:
            diff = height / 2
            right_space = width - x_local
            if right_space < diff:
                left_adjust = 2 * diff - right_space
                cutted_image = image[0:height, x_local - left_adjust:width]
            else:
                right_adjust = 2 * diff - x_local
                cutted_image = image[0:height, 0:x_local + right_adjust]
        else:
            diff = width / 2
            top_space = height - y_local
            if top_space < diff:
                bot_adjust = 2 * diff - top_space
                cutted_image = image[y_local - bot_adjust:height, 0:width]
            else:
                top_adjust = 2 * diff - y_local
                cutted_image = image[0:y_local + top_adjust, 0:width]
        return cutted_image

    def __simple_cut(self, image):
        height = image.shape[0]
        width = image.shape[1]
        x_local = width / 2
        y_local = height / 2
        if height < width:
            diff = height / 2
            cutted_image = image[0:height, x_local - diff:x_local + diff]
        else:
            diff = width/2
            cutted_image = image[y_local - diff:y_local + diff, 0:width]
        return cutted_image

    def available_images(self):
        return self.image_dir

    def __init_cache_dir(self):
        if not os.path.exists(self.cache_dir_path):
            os.mkdir(self.cache_dir_path)
        else:
            self.cache_dir = os.listdir(self.cache_dir_path)

    def __init_image_dir(self):
        image_names = []
        all_files = os.listdir(self.image_dir_path)
        for some_file in all_files:
            filext = os.path.splitext(some_file)[1]
            if filext in self.image_ext:
               image_names.append(some_file)
        return image_names







