# pylint: disable= invalid-name, duplicate-code
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
        cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''',
                               (row[0], "pre"))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(19)
    util.setup("LGG_PRE/")

    pre_images = find_images()
    image_registration.get_transforms(pre_images, image_registration.SYN, save_to_db=True)
