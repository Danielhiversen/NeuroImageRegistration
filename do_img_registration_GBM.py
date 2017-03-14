# pylint: disable= invalid-name
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""
from __future__ import print_function
import datetime
import os
import sqlite3

import image_registration
import util


def find_images():
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient where study_id = ?''', ("qol_grade3,4", ))
    ids = []
    k = 0
    for row in cursor:
        k += 1
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (row[0], ))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    print(ids, k)
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("GBM_" + "{:%m_%d_%Y}_BE2".format(datetime.datetime.now()) + "/")

    moving_datasets_ids = find_images()
    print(len(moving_datasets_ids))

    data_transforms = image_registration.get_transforms(moving_datasets_ids,
                                                        image_registration.SYN,
                                                        save_to_db=True)

#    image_registration.save_transform_to_database(data_transforms)
