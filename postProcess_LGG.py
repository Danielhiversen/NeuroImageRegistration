# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
from os.path import basename
import nipype.interfaces.ants as ants
import datetime
import sqlite3

from img_data import img_data
import util


def find_images(diag):
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient where study_id = ? ''',
                          ("LGG_reseksjonsgrad", ))
    ids = []
    k = 0
    for row in cursor:
        k += 1
        cursor2 = conn.execute('''SELECT id from Images where pid = ? and diag_pre_post = ?''',
                               (row[0],  diag))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


def post_calculations(moving_dataset_image_ids, result=None):
    """ Transform images and calculate avg"""
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    conn.text_factory = str
    if result is None:
        result = {}

    for _id in moving_dataset_image_ids:
        img = img_data(_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img.load_db_transforms()

        reg_vol = util.transform_volume(img.img_filepath, img.get_transforms())
        vol = util.TEMP_FOLDER_PATH + util.get_basename(basename(reg_vol)) + '_BE.nii.gz'

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg_vol
        mult.inputs.second_input = util.TEMPLATE_MASK
        mult.inputs.output_product_image = vol
        mult.run()

        label = "img"
        if label in result:
            result[label].append(vol)
        else:
            result[label] = [vol]

        for (segmentation, label) in util.find_reg_label_images(_id):
            segmentation = util.transform_volume(segmentation, img.get_transforms())
            if label in result:
                result[label].append(segmentation)
            else:
                result[label] = [segmentation]
    return result


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    image_ids = find_images("pre")
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all_pre', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img_pre', None, True, folder)

    image_ids = find_images("post")
    result = post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all_post', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img_post', None, True, folder)


if __name__ == "__main__":
    folder = "RES_LGG_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"
    process(folder)
