import random
import cProfile
from src.image_loader.image_loader import ImageLoader
from src.feature_extractors.texture_feature_extractor import TextureFeatureExtractor


def generate_new_feature_extractor():
    il = ImageLoader(image_dir_path='../../../data/train')
    cats_images = ['cat.{}.jpg'.format(i) for i in range(1000)]
    dogs_images = ['dog.{}.jpg'.format(i) for i in range(1000)]
    images_for_tailing = cats_images + dogs_images
    random.shuffle(images_for_tailing)
    tfe = TextureFeatureExtractor(il, images_for_tailing_names=images_for_tailing, tiles_count=1000, delta=0.18, debug=True)
    used_images = tfe.generate_tiles()
    print('used images: {}'.format(used_images))
    tfe.show_tiles()
    tfe.save('texture_feature_extractor_delta=018')


def generate_cache():
    il = ImageLoader(image_dir_path='../../../data/train')
    tfe = TextureFeatureExtractor.load('texture_feature_extractor_delta=018')
    images_for_caching = il.available_images()
    for num, image_name in enumerate(images_for_caching):
        img = il.load(image_name)
        tfe.extract(img)
        print('progress: {:.2f}%'.format(float(num + 1) / len(images_for_caching) * 100))
    tfe.save('texture_feature_extractor_cache_delta=018')


def test_cache():
    il = ImageLoader(image_dir_path='../../../data/train')
    tfe = TextureFeatureExtractor.load('texture_feature_extractor_cache_delta=018')
    for image_name in il.available_images():
        img = il.load(image_name)
        print(tfe.extract(img))


if __name__ == '__main__':
    # generate_new_feature_extractor()
    generate_cache()
    # cProfile.run('test_cache()')