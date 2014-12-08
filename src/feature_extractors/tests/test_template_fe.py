import time


from src.image_loader.image_loader import ImageLoader
from src.feature_extractors.template_feature_extractor import TemplateFeatureExtractor

import cProfile

def test():
    il = ImageLoader(image_dir_path='../../../data/train')
    tfe = TemplateFeatureExtractor(il, tiles_count=1000, delta=0.5, debug=True)
    start = time.time()
    used_images = tfe.generate_tiles()
    print('used images: {}'.format(used_images))
    stop = time.time()
    tfe.show_tiles()
    print('tile generation: {}'.format(stop - start))
    img = il.load(il.available_images()[100])
    print('features: {}'.format(tfe.extract(img)))


def main():
    use_cprofile = False
    if use_cprofile:
        cProfile.run('test()')
    else:
        test()


if __name__ == "__main__":
    main()