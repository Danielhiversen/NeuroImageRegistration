# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""

import os
import sys

import image_registration


def process_dataset(post_image):
    """ pre process and registrate volume"""
    num_tries = 3
    pre_image = post_image.replace("post", "pre")
    for k in range(num_tries):
        try:
            pre_image_pre = image_registration.pre_process(pre_image, False)
            post_image_pre = image_registration.pre_process(post_image, False)
            trans1 = image_registration.registration(post_image_pre, pre_image_pre,
                                                     image_registration.RIGID)

            pre_image_pre2 = image_registration.pre_process(pre_image)
            trans2 = image_registration.registration(pre_image_pre2,
                                                     image_registration.TEMP_FOLDER_PATH +
                                                     "masked_template.nii",
                                                     image_registration.AFFINE)
            return (post_image, trans1, trans2)
        # pylint: disable=  broad-except
        except Exception as exp:
            print('Crashed during processing of ' + post_image + '. Try ' +
                  str(k+1) + ' of ' + str(num_tries) + ' \n' + str(exp))


# pylint: disable= invalid-name
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python " + __file__ + " dataset")
        exit(1)
    os.nice(19)
    image_registration.setup(sys.argv[1:])
    if not os.path.exists(image_registration.TEMP_FOLDER_PATH):
        os.makedirs(image_registration.TEMP_FOLDER_PATH)
    if not os.path.exists(image_registration.DATA_OUT_PATH):
        os.makedirs(image_registration.DATA_OUT_PATH)

    image_registration.prepare_template(image_registration.TEMPLATE_VOLUME,
                                        image_registration.TEMPLATE_MASK)

    post_images = image_registration.find_moving_images('/home/dahoiv/disk/data/LGG_kart/POST/')

    data_transforms = image_registration.move_dataset(post_images, process_dataset)
    results = image_registration.move_segmentations(data_transforms)

    for label_i in results:
        image_registration.post_calculation(results[label_i], label_i)

    if sys.argv[1] == "test":
        print(len(results))
        print(len(data_transforms))
        print(len(results))
