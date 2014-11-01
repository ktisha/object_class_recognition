__author__ = 'avesloguzova'
import cv2

# General params
base_path = ""

# Params of training samples
first_class_params = dict(prefix="cat.",
                          count=12499,
                          train_count=200,
                          test_count=100,
                          postfix=".jpg")

second_class_params = dict(prefix="dog.",
                           count=12499,
                           train_count=200,
                           test_count=100,
                           postfix=".jpg")


# Params of SVM
svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC,
                  C=2.67,
                  gamma=5.383)

svm_save_file = "svm_classifier.dat"

# Params of image loader
image_dir = "../data/train"

# Params of texture_feature_extractor
tile_size = 50
tiles_count = 1
images_for_tailing_count = 500
delta = 80.0