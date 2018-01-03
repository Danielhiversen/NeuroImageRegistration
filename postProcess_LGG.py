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


def find_images(diag_pre_post):
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
                               (row[0],  diag_pre_post))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


def post_calculations(moving_dataset_image_ids, result=None):
    """ Transform images and calculate avg"""
    if result is None:
        result = {}

    for _id in moving_dataset_image_ids:
        img = img_data(_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img.load_db_transforms()

        img_pre = img_data(img.fixed_image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img_pre.load_db_transforms()

        reg_vol = util.transform_volume(img.reg_img_filepath, img_pre.get_transforms())
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
            segmentation = util.transform_volume(segmentation, img_pre.get_transforms(), label_img=True)
            if label in result:
                result[label].append(segmentation)
            else:
                result[label] = [segmentation]
    return result


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)

    image_ids = find_images("post")
    if True:
        import nibabel as nib
        conn = sqlite3.connect(util.DB_PATH)
        conn.text_factory = str
        cursor = conn.execute(
            '''SELECT pid from Patient where study_id = ? ''',
            ("LGG_reseksjonsgrad",))
        k = 0
        for row in cursor:
            k += 1
            cursor2 = conn.execute(
                '''SELECT id from Images where pid = ? and diag_pre_post = ?''',
                (row[0], "post"))
            ids = []
            for _id in cursor2:
                ids.append(_id[0])
            cursor2.close()
            if not ids:
                continue
            images_post = post_calculations(ids)
            file_name_post = images_post['all'][0]

            cursor2 = conn.execute(
                '''SELECT id from Images where pid = ? and diag_pre_post = ?''',
                (row[0], "pre"))
            ids = []
            for _id in cursor2:
                ids.append(_id[0])
            images_pre = util.post_calculations(ids)
            file_name_pre = images_pre['all'][0]

            img_pre = nib.load(file_name_pre)
            img_post = nib.load(file_name_post)
            temp = img_pre.get_data() - img_post.get_data()
            temp = temp.flatten()
            print(sum(temp < 0), file_name_pre, file_name_post)

        cursor.close()
        conn.close()
        return

    result_post = post_calculations(image_ids)
    print(len(result_post['all']))
    util.avg_calculation(result_post['all'], 'all_post', None, True, folder, save_sum=True)
    util.avg_calculation(result_post['img'], 'img_post', None, True, folder)

    image_ids = find_images("pre")
    result_pre = util.post_calculations(image_ids)
    print(len(result_pre['all']))
    util.avg_calculation(result_pre['all'], 'all_pre', None, True, folder, save_sum=True)
    util.avg_calculation(result_pre['img'], 'img_pre', None, True, folder)
    util.calc_resection_prob(result_pre['all'], result_post['all'], 'resection_prob', True, folder, -1)


if __name__ == "__main__":
    folder = "RES_LGG_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"
    process(folder)
