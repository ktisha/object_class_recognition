__author__ = 'Andrey Lazarevich'
import cv2
import numpy
import os


def nested_rect(candidate, bound):
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
    def __init__(self, image_dir_path=os.getcwd(), cut_size=250, haarcascade_path='haarcascade_eye.xml'):
        self.haarcascade_name = haarcascade_path
        self.min_cont_len = 100
        self.cascade = []
        self.__init_haar()
        assert isinstance(cut_size, int)
        self.cut_size = cut_size
        self.image_ext = [".jpg", ".png", ".bmp"]
        self.image_dir_path = image_dir_path
        self.image_dir = self.__init_image_dir()
        self.cache_dir_path = os.path.join('../data', 'cache')
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
            full_path = os.path.join(self.image_dir_path, image_name)
            image = cv2.imread(full_path)
            if not self.cascade:
                cutted_image = self.__simple_cut(image)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # bound_rects = get_rects(gray)
                # x_local, y_local = self.__find_approx_center_in_rect_by_eyes(gray, bound_rects)
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
        if image_name in self.cache_dir:
            image = cv2.imread(os.path.join(self.cache_dir_path, image_name))
            return image
        else:
            raise IndexError

    def __smart_cut(self, x_local, y_local, image):
        height = image.shape[0]
        width = image.shape[1]
        if height < self.cut_size and width < self.cut_size:
            cutted_image = self.__simple_cut(image)
        else:
            if height < width:
                diff = height / 2
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
        height = image.shape[0]
        width = image.shape[1]
        x_local = width / 2
        y_local = height / 2
        if height < width:
            diff = height / 2
            cutted_image = image[0:height, x_local - diff:x_local + diff]
        else:
            diff = width / 2
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

    def __find_approx_center_in_rect_by_eyes(self, gray, bound_rects):
        x_local = 0
        y_local = 0
        valid_rects = 0
        for bound_rect in bound_rects:
            search_area = gray[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]
            eyes = self.cascade.detectMultiScale(search_area)
            try:
                x_haar = eyes[0][0]
                y_haar = eyes[0][1]
                w_haar = eyes[0][2]
                h_haar = eyes[0][3]
                x_local += (x_haar + x_haar + w_haar) / 2 + bound_rect[0]
                y_local += (y_haar + y_haar + h_haar) / 2 + bound_rect[1]
                valid_rects += 1
            except IndexError:
                pass
        if valid_rects:
            x_local /= valid_rects
            y_local /= valid_rects
        return x_local, y_local

    def __find_approx_center_in_conts_by_mass_center(self, conts):
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
        center_x /= len(centers)
        center_y /= len(centers)
        return center_x, center_y