# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:51:02 2016

@author: dahoiv
"""

import os
import datetime
import sqlite3

import image_registration
import util


def find_images():
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')
    ids = []
    for row in cursor:
        cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''', (row[0], "pre"))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("GBM_LGG_TEMP_" + "{:%m_%d_%Y}".format(datetime.datetime.now()) + "/")

    image_registration.prepare_template(util.TEMPLATE_VOLUME, util.TEMPLATE_MASK)

    moving_datasets_ids = find_images()
    print(moving_datasets_ids)
    data_transforms = image_registration.get_transforms(moving_datasets_ids, image_registration.SYN)

#    image_registration.save_transform_to_database(data_transforms)
