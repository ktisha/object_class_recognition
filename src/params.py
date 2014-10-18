__author__ = 'avesloguzova'
import cv2


svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC,
                  C=2.67,
                  gamma=5.383)
svm_save_file = "svm_classifier.dat"
image_dir = "../data/train"