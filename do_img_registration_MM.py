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


def find_images(exclude=None):
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

#            _img_filepath = cursor3.fetchone()[0]
#            if _img_filepath and os.path.exists(util.DATA_FOLDER + _img_filepath):
#                cursor3.close()
#                continue
            if exclude and _id in exclude:
                continue
            ids.append(_id)
            cursor3.close()

        cursor2.close()

    cursor.close()
    conn.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":
    os.nice(17)
    util.setup("MM_TEMP_" + "{:%m_%d_%Y}_BE2".format(datetime.datetime.now()) + "/", "MolekylareMarkorer")

    moving_datasets_ids_affine = [7, 39, 31]
#    moving_datasets_ids = find_images(exclude=moving_datasets_ids_affine)
#    print(moving_datasets_ids, len(moving_datasets_ids))
#    data_transforms = image_registration.get_transforms(moving_datasets_ids,
#                                                        image_registration.SYN,
#                                                        save_to_db=True)

    print(moving_datasets_ids_affine, len(moving_datasets_ids_affine))
    data_transforms = image_registration.get_transforms(moving_datasets_ids_affine,
                                                        image_registration.RIGID,
                                                        save_to_db=True, be_method=0)
