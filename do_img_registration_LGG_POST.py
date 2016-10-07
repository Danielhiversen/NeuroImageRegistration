# pylint: disable= invalid-name
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""
from __future__ import print_function
import os
import sqlite3

from img_data import img_data
import image_registration
import util


def find_images():
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')
    ids = []
    for row in cursor:
        cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''',
                               (row[0], "post"))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    print(ids)
    return ids


def process_dataset(args, num_tries=3):
    """ pre process and registrate volume"""
    # pylint: disable= unused-argument
    moving_image_id = args[0]
    print(moving_image_id)

    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Images where id = ?''', (moving_image_id,))
    pid = cursor.fetchone()[0]
    cursor = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''',
                          (pid, "pre"))
    db_temp = cursor.fetchone()
    pre_image_id = db_temp[0]

    cursor.close()
    conn.close()

    import datetime
    start_time = datetime.datetime.now()
    pre_img = img_data(pre_image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
    post_img = img_data(moving_image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)

    pre_img = image_registration.pre_process(pre_img, False)
    post_img = image_registration.pre_process(post_img, False)
    img = image_registration.registration(post_img, pre_img.pre_processed_filepath,
                                          image_registration.RIGID)
    print("\n\n\n\n -- Total run time: ")
    print(datetime.datetime.now() - start_time)

    img.fixed_image = pre_image_id

    return img


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("LGG_POST/", "LGG")

    post_images = find_images()
    data_transforms = image_registration.get_transforms(post_images,
                                                        process_dataset_func=process_dataset,
                                                        save_to_db=True)

#    image_registration.save_transform_to_database(data_transforms)
