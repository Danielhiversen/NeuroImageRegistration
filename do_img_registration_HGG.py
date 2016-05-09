# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""

import os
import sys
import sqlite3


import image_registration

DATA_FOLDER = "/mnt/dokumneter/data/test/"
DB_PATH = DATA_FOLDER + "brainSegmentation.db"


def find_images_qol():
    """ Find images for registration """
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from QualityOfLife''')
    images = []
    for row in cursor:
        cursor2 = conn.execute('''SELECT filepath from Images where pid = ? ''', (row[0],))
        images.append(DATA_FOLDER + cursor2.fetchone()[0])
        cursor2.close()

    cursor.close()
    conn.close()
    print(images)
    return images


def find_seg_images(moving):
    """ Find segmentation images"""
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT id from Images where filepath = ? ''', (moving,))
    image_id = cursor.fetchone()[0]
    cursor2 = conn.execute('''SELECT filepath from Labels where image_id = ? ''', (image_id,))
    images = []
    for row in cursor2:
        images.append(DATA_FOLDER + row)

    cursor.close()
    cursor2.close()
    conn.close()
    return images


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    image_registration.setup(["HGG"])
    if not os.path.exists(image_registration.TEMP_FOLDER_PATH):
        os.makedirs(image_registration.TEMP_FOLDER_PATH)
    if not os.path.exists(image_registration.DATA_OUT_PATH):
        os.makedirs(image_registration.DATA_OUT_PATH)

    image_registration.prepare_template(image_registration.TEMPLATE_VOLUME,
                                        image_registration.TEMPLATE_MASK)

    moving_datasets = find_images_qol()

    data_transforms = image_registration.move_dataset(moving_datasets)
    results = image_registration.move_segmentations(data_transforms)

    for label_i in results:
        image_registration.post_calculation(results[label_i], label_i)

    if sys.argv[1] == "test":
        print(len(results))
        print(len(data_transforms))
        print(len(results))
