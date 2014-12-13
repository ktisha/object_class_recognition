#!/usr/bin/env python3
__author__ = 'Yurgin Pavel'

import os

from src.feature_extractors.feature_extractor import FeatureExtractor
import numpy as np
from decaf.scripts.imagenet import DecafNet


class ConvNetFeatureExtractor(FeatureExtractor):
    def __init__(
            self,
            feature_layer='fc7_cudanet_out',
            pretrained_params='imagenet.decafnet.epoch90',
            pretrained_meta='imagenet.decafnet.meta',
            center_only=True
            ):
        """
        :param feature_layer: The ConvNet layer that's used for
                              feature extraction.  Defaults to
                              `fc7_cudanet_out`.  A description of all
                              available layers for the
                              ImageNet-1k-pretrained ConvNet is found
                              in the DeCAF wiki.  They are:

                                - `pool5_cudanet_out`
                                - `fc6_cudanet_out`
                                - `fc6_neuron_cudanet_out`
                                - `fc7_cudanet_out`
                                - `fc7_neuron_cudanet_out`

        :param pretrained_params: This must point to the file with the
                                  pretrained parameters.  Defaults to
                                  `imagenet.decafnet.epoch90`.  For
                                  the ImageNet-1k-pretrained ConvNet
                                  this file can be obtained from here:
                                  http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/

        :param pretrained_meta: Similar to `pretrained_params`, this
                                must file to the file with the
                                pretrained parameters' metadata.
                                Defaults to `imagenet.decafnet.meta`.

        :param center_only: Use the center patch of the image only
                            when extracting features.  If `False`, use
                            four corners, the image center and flipped
                            variants and average a total of 10 feature
                            vectors, which will usually yield better
                            results.  Defaults to `True`.
        """
        super(ConvNetFeatureExtractor, self).__init__()
        self.feature_layer = feature_layer
        self.pretrained_params = pretrained_params
        self.pretrained_meta = pretrained_meta
        self.center_only = center_only
        self.convnet = DecafNet(
            self.pretrained_params,
            self.pretrained_meta
        )

    def _extract(self, img):
        """
        :param cv2 image
        :return: np.array with shape (4096,)
        """
        img = self.convnet.oversample(img, center_only=self.center_only)
        self.convnet.classify_direct(img)
        feat = self.convnet.feature(self.feature_layer)
        if not self.center_only:
            feat = feat.mean(0)
        return feat[0]


if __name__ == "__main__":
    pass