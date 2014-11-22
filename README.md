object_class_recognition
========================

SpbAU NIR 2014

The training archive contains 25,000 images of dogs and cats. It's quite a big archive so it's splitted into parts.
Use 
zip -F data.zip --out train.zip 
to recreate full train archive.

Deployment:

0. Install OpenCV use https://github.com/ktisha/object_class_recognition/wiki/%D0%9D%D0%B0%D1%81%D1%82%D1%80%D0%BE%D0%B9%D0%BA%D0%B0-%D0%BE%D0%BA%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F
1. `source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `./build.sh`
4. Unpack dataset
5. Run `src/feature_extractors/tests/generate_texture_cache.py`
6. Wait about 4 days. In console there are progress bar.
7. Copy file `texture_feature_extractor_cache_delta=018` for example to https://drive.google.com/?tab=mo&authuser=0#folders/0Byzih9QxjDRmUGRieDZrYldoSVU
