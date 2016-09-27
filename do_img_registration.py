# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:51:02 2016

@author: dahoiv
"""
from __future__ import print_function
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
        cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''',
                               (row[0], "pre"))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


def find_images_with_qol():
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str

    cursor = conn.execute('''SELECT pid from QualityOfLife''')
    pids_with_qol = []
    for row in cursor:
        if row:
            pids_with_qol.append(row[0])
    cursor.close()

    cursor = conn.execute('''SELECT pid from Patient''')
    ids = []
    for row in cursor:
        pid = row[0]
        if pid not in pids_with_qol:
            continue
        cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''',
                               (pid, "pre"))

        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("GBM_LGG_TEMP_" + "{:%m_%d_%Y}_BE2".format(datetime.datetime.now()) + "/")

    moving_datasets_ids = find_images_with_qol()
    print(moving_datasets_ids)
    data_transforms = image_registration.get_transforms(moving_datasets_ids, image_registration.SYN,
                                                        saveToDb=True)

    # image_registration.save_transform_to_database(data_transforms)
