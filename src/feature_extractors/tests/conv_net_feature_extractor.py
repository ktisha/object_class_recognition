#!/usr/bin/env python3
__author__ = 'Yurgin Pavel'

from src.image_loader.image_loader import ImageLoader
from src.feature_extractors.feature_extractor import FeatureExtractor
from src.feature_extractors.conv_net_feature_extractor import ConvNetFeatureExtractor
from src.feature_extractors.color_feature_extractor import ColorFeatureExtractor

def test_extract():
    from time import clock
    i_loader = ImageLoader(image_dir_path='../../../data/train')
    available_images = i_loader.available_images()
    feature_extractor = ConvNetFeatureExtractor(
        pretrained_meta="/home/noname/nir/venv/cat_vs_dog/data/pretrain/imagenet.decafnet.meta",
        pretrained_params="/home/noname/nir/venv/cat_vs_dog/data/pretrain/imagenet.decafnet.epoch90"
    )

    t = clock()
    for img in available_images[:100]:
        image = i_loader.load(img)
        feature_extractor.extract(image)
    print(clock() - t)


if __name__ == "__main__":
    test_extract()