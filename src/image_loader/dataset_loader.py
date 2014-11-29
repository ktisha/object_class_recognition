import cv2
import os
from pycocotools.coco import COCO


class DatasetLoader:
    def __init__(self, images="../data/COCO/images", annotations="../data/COCO/annotations/instances_train2014.json"):
        """

        :param images: directory to COCO dataset
        :param annotations: path to description file
        """
        self.dataset_dir = os.path.join(images, 'datasets')
        self.coco = COCO(annotations, images)
        self.animals_ids = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22,
                            'bear': 23, 'zebra': 24, 'giraffe': 25}
        # just a hardcoded numbers. nothing special.

    def get_dataset(self, category_name='cat', number_of_elements=500):
        """

        :param category: animal category. can be easily taken from animals_ids
        :param number_of_elements: number of elements to return
        """
        counter = 0
        names_list = []
        category = self.animals_ids[category_name]
        im_id_list = self.coco.getImageIds(params={'cat_id': category})
        anns = self.coco.loadAnnotations(params={'im_id_list': im_id_list, 'cat_id:': category, 'area__gt': 150*150})
        for ann in anns:
            if counter > number_of_elements:
                break
            id = ann['image_id']
            name = self.coco.images[id]['file_name']
            path = self.coco.images[id]['file_path']
            new_name = category_name + '.' + str(counter) + '.jpg'
            img = cv2.imread(os.path.join(self.coco.image_folder, path, name))
            cv2.imwrite(os.path.join(self.dataset_dir, new_name), img)
            names_list.append(new_name)
            counter += 1
        return names_list
