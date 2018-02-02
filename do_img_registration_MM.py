# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:51:02 2016

@author: dahoiv
"""
from __future__ import print_function
import os
import datetime
import nibabel as nib
import sqlite3
from nilearn import datasets

import image_registration
import util


def find_images(exclude=None, max_dim_size=None):
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str

    cursor = conn.execute('''SELECT pid from Patient''')
    ids = []
    for row in cursor:
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (row[0], ))
        for _id in cursor2:
            _id = _id[0]
            cursor3 = conn.execute('''SELECT filepath_reg from Images where id = ? ''', (_id,))

            # _img_filepath = cursor3.fetchone()[0]
            cursor3.close()
            # if _img_filepath and os.path.exists(util.DATA_FOLDER + _img_filepath):
            #     continue
            if exclude and _id in exclude:
                continue
            if max_dim_size:
                _filepath = conn.execute('''SELECT filepath from Images where id = ?''', (_id,)).fetchone()[0]
                img_shape = nib.load(util.DATA_FOLDER + _filepath).get_data().shape
                if img_shape[0] > max_dim_size and img_shape[1] > max_dim_size and img_shape[2] > max_dim_size:
                    continue
            ids.append(_id)

        cursor2.close()

    cursor.close()
    conn.close()
    return ids


def find_images_from_pid(pids, exclude=None):
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str

    ids = []
    for pid in pids:
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (pid, ))
        for _id in cursor2:
            _id = _id[0]
            if exclude and _id in exclude:
                continue
            ids.append(_id)
        cursor2.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    HOSTNAME = os.uname()[1]
    if 'unity' in HOSTNAME or 'compute' in HOSTNAME:
        path = "/work/danieli/MM/"
    else:
        os.nice(17)
        path = "MM_TEMP_" + "{:%m_%d_%Y}_BE2".format(datetime.datetime.now()) + "/"

    util.setup(path, "MolekylareMarkorer")
    #
    # util.prepare_template(datasets.fetch_icbm152_2009(data_dir="./").get("t2"), util.TEMPLATE_MASK, True)
    # moving_datasets_ids = find_images_from_pid([164])
    # print(moving_datasets_ids, len(moving_datasets_ids))
    # data_transforms = image_registration.get_transforms(moving_datasets_ids,
    #                                                     image_registration.AFFINE,
    #                                                     save_to_db=True)

    util.prepare_template(datasets.fetch_icbm152_2009(data_dir="./").get("t2"), util.TEMPLATE_MASK, True)
    moving_datasets_ids = find_images_from_pid([125, 2101])
    print(moving_datasets_ids, len(moving_datasets_ids))
    data_transforms = image_registration.get_transforms(moving_datasets_ids,
                                                        image_registration.AFFINE,
                                                        save_to_db=True)

    # moving_datasets_ids = find_images_from_pid([172])
    # print(moving_datasets_ids, len(moving_datasets_ids))
    # data_transforms = image_registration.get_transforms(moving_datasets_ids,
    #                                                     image_registration.RIGID,
    #                                                     save_to_db=True)

    # print(moving_datasets_ids_affine, len(moving_datasets_ids_affine))
    # data_transforms = image_registration.get_transforms(moving_datasets_ids_affine,
    #                                                     image_registration.RIGID,
    #                                                     save_to_db=True, be_method=0)
