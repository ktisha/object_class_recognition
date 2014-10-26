from src.image_loader.image_loader import ImageLoader

from texture_feature_extractor_for_debug import TextureFeatureExtractor


il = ImageLoader(image_dir_path='../../../data/train')
tfe = TextureFeatureExtractor(il)
tfe._debug_generate_tiles()
img = il.load(il.available_images()[0])

def f0():
    f_0 = tfe.extract(img)
    print(f_0)

def f1():
    f_1 = tfe.extract_1(img)
    print(f_1)
def f2():
    f_2 = tfe.extract_2(img)
    print(f_2)
def f3():
    f_3 = tfe.extract_3(img)
    print(f_3)

from timeit import timeit

n = 1

print(timeit('f0()', setup='from __main__ import f0', number=n))
print(timeit('f1()', setup='from __main__ import f1', number=n))
print(timeit('f2()', setup='from __main__ import f2', number=n))
print(timeit('f3()', setup='from __main__ import f3', number=n))

#cProfile.run('f2()')
