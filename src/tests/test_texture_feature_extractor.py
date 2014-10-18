import time

import cv2
from src.image_loader import ImageLoader
from src.texture_feature_extractor import TextureFeatureExtractor

import cProfile

def test():
    il = ImageLoader(image_dir_path='../data/train')
    tfe = TextureFeatureExtractor(il)
    tfe.generate_tiles()
    img = il.load(il.available_images()[0])
    start = time.time()
    print(tfe.extract(img))
    stop = time.time()
    print('elapsed time: {}'.format(stop - start))
    #cv2.imshow('test_tile_tile_distance', img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()


def main():
    use_cprofile = True
    if use_cprofile:
        cProfile.run('test()')
    else:
        test()


if __name__ == "__main__":
    main()