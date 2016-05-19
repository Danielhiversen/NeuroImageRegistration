# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""

import os
import sqlite3
import ConvertDataToDB
import image_registration


def find_images_lgg_post():
    """ Find images for registration """
    conn = sqlite3.connect(image_registration.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient where diagnose = ?''', ('LGG',))
    ids = []
    for row in cursor:
        cursor2 = conn.execute('''SELECT id, transform from Images where pid = ? AND diag_pre_post = ?''', (row[0], "post"))
        for (_id, _transform) in cursor2:
            if _transform is not None:
                continue
            ids.append(_id)
        cursor2.close()

    cursor.close()
    conn.close()
    print(ids)
    return ids


def process_dataset(args, num_tries=3):
    """ pre process and registrate volume"""
    (moving_image_id, reg_type) = args
    conn = sqlite3.connect(image_registration.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath from Images where id = ? ''', (moving_image_id,))
    post_image = image_registration.DATA_FOLDER + cursor.fetchone()[0]

    cursor = conn.execute('''SELECT pid from Images where id = ?''', (moving_image_id,))
    pid = cursor.fetchone()[0]
    cursor = conn.execute('''SELECT filepath from Images where pid = ? AND diag_pre_post = ?''', (pid, "pre"))
    pre_image = image_registration.DATA_FOLDER + cursor.fetchone()[0]
    print(pre_image, post_image)

    cursor.close()
    conn.close()

    for k in range(num_tries):
        pre_image_pre = image_registration.pre_process(pre_image, False)
        post_image_pre = image_registration.pre_process(post_image, False)
        trans1 = image_registration.registration(post_image_pre, pre_image_pre,
                                                 image_registration.RIGID)

        pre_image_pre2 = image_registration.pre_process(pre_image)
        trans2 = image_registration.registration(pre_image_pre2,
                                                 image_registration.TEMP_FOLDER_PATH +
                                                 "masked_template.nii",
                                                 image_registration.AFFINE)
        try:
            return (post_image, [trans2, trans1])
        # pylint: disable=  broad-except
        except Exception as exp:
            raise Exception('Crashed during processing of ' + post_image + '. Try ' +
                            str(k+1) + ' of ' + str(num_tries) + ' \n' + str(exp))


def move_segmentations(transforms):
    """ move label image with transforms """
    result = dict()
    for moving, transform in transforms:
        for segmentation in image_registration.find_seg_images(moving):
            temp = image_registration.move_data(segmentation, transform)
            label = image_registration.find_label(temp)
            if label in result:
                result[label].append(temp)
            else:
                result[label] = [temp]
    return result

# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    image_registration.setup("LGG_POST/")
    if not os.path.exists(image_registration.TEMP_FOLDER_PATH):
        os.makedirs(image_registration.TEMP_FOLDER_PATH)

    image_registration.prepare_template(image_registration.TEMPLATE_VOLUME,
                                        image_registration.TEMPLATE_MASK)

    post_images = find_images_lgg_post()

    data_transforms = image_registration.get_transforms(post_images, process_dataset_func=process_dataset)

    ConvertDataToDB.save_transform_to_database(data_transforms)
    