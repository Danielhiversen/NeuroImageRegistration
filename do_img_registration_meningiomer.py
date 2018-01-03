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
import sys

import image_registration
import util


def find_images(pids=[], exclude=[]):
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')
    ids = []
    for row in cursor:
        if pids and row[0] not in pids:
            continue
        if row[0] in exclude:
            continue
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (row[0], ))
        for _id in cursor2:
            _id = _id[0]
            cursor3 = conn.execute('''SELECT filepath_reg from Images where id = ? ''', (_id,))

            cursor3.close()
            ids.append(_id)

    cursor.close()
    conn.close()
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":  # if 'unity' in hostname or 'compute' in hostname:
    os.nice(19)
    HOSTNAME = os.uname()[1]
    if 'unity' in HOSTNAME or 'compute' in HOSTNAME:
        path = "/work/danieli/meningiomer/"
    else:
        path = "meningiomer_" + "{:%m_%d_%Y}".format(datetime.datetime.now()) + "/"

    util.setup(path, "meningiomer")

    moving_datasets_ids = find_images(pids=[1, 35, 105, 192, 201, 463, 508], exclude=[])

    if len(sys.argv) > 2:
        num_of_splits = int(sys.argv[1])
        split = int(sys.argv[2])

        length = int(len(moving_datasets_ids) / num_of_splits)
        start_idx = length * (split - 1)
        if split < num_of_splits:
            moving_datasets_ids = moving_datasets_ids[start_idx:(start_idx+length)]
        else:
            moving_datasets_ids = moving_datasets_ids[start_idx:]

    util.LOGGER.info(str(moving_datasets_ids) + " " + str(len(moving_datasets_ids)))
    image_registration.BET_FRAC = 0.2
    image_registration.get_transforms(moving_datasets_ids,
                                      reg_type=image_registration.COMPOSITEAFFINE,
                                      reg_type_be=image_registration.COMPOSITEAFFINE,
                                      save_to_db=True)

    # moving_datasets_ids = find_images([1052, 1167, 1, 31, 35, 192, 201, 388, 397, 463, 508, 530, 563, 709, 866, 927, 941, 981, 1020])
    # #excluded: 1061, 709
    # print(moving_datasets_ids)
    # image_registration.get_transforms(moving_datasets_ids,
    #                                   image_registration.AFFINE,
    #                                   reg_type_be=image_registration.COMPOSITEAFFINE,
    #                                   save_to_db=True)


# 192, 201, 35, 463. 709, 1020



# 1, 35, 105, 192, 201, 463, 508, 530, 563, 709, 866, 1020, 1061,  image_registration.BET_FRAC = 0.1

# 1, 35,      192, 201, 463, 508, 530, 709, 866, 1020, 1061,  image_registration.BET_FRAC = 0.3


# 508, 866,

