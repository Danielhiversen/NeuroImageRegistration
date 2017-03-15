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
    return ids


# pylint: disable= invalid-name
if __name__ == "__main__":  # if 'unity' in hostname or 'compute' in hostname:
    os.nice(19)
    util.setup("GBM_" + "{:%m_%d_%Y}".format(datetime.datetime.now()) + "/")

    moving_datasets_ids = find_images()

    if len(sys.argv) > 2:
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # split = comm.Get_rank()
        # num_of_splits = comm.Get_size()

        num_of_splits = int(sys.argv[1])
        split = int(sys.argv[2])

        length = int(len(moving_datasets_ids) / num_of_splits)
        start_idx = length * (split - 1)
        if split < num_of_splits:
            moving_datasets_ids = moving_datasets_ids[start_idx:(start_idx+length)]
        else:
            moving_datasets_ids = moving_datasets_ids[start_idx:]

    util.LOGGER.info(str(moving_datasets_ids) + " " + str(len(moving_datasets_ids)))
    image_registration.get_transforms(moving_datasets_ids,
                                      image_registration.SYN,
                                      save_to_db=True)
