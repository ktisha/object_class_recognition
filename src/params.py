__author__ = 'avesloguzova'
import cv2

# General params
base_path = ""
logfile = "../result.log"
features_cache = "../data/feature_extractor_cache/color_feature_cacche_5_10_6_6.gz"

# Params of training samples
class_params = [dict(name="cat",
                     label=1,
                     sample_size=3000),

                dict(name="dog",
                     label=2,
                     sample_size=3000),

                dict(name="horse",
                     label=3,
                     sample_size=3000)]


# Params of SVM
svm_params = dict(kernel_type=cv2.SVM_RBF,
                  svm_type=cv2.SVM_C_SVC,
                  C=100,
                  gamma=0.001)

svm_save_file = "svm_classifier.dat"

# Params of image loader
image_dir = "../data/train"


# Params of texture_feature_extractor
tile_size = 50
tiles_count = 1
images_for_tailing_count = 500
delta = 80.0