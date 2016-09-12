# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""

import os
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
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (row[0], ))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    print(ids)
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("GBM/", "GBM")
    if not os.path.exists(util.TEMP_FOLDER_PATH):
        os.makedirs(util.TEMP_FOLDER_PATH)

    image_registration.prepare_template(image_registration.TEMPLATE_VOLUME,
                          image_registration.TEMPLATE_MASK)

    moving_datasets_pids = find_images()

    data_transforms = image_registration.get_transforms(moving_datasets_pids, image_registration.SYN)

    image_registration.save_transform_to_database(data_transforms)
