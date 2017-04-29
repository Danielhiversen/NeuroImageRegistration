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


# pylint: disable= invalid-name
if __name__ == "__main__":  # if 'unity' in hostname or 'compute' in hostname:
    HOSTNAME = os.uname()[1]
    if 'unity' in HOSTNAME or 'compute' in HOSTNAME:
        path = "/work/danieli/GBM_survival/"
    else:
        os.nice(19)
        path = "GBM_" + "{:%m_%d_%Y}".format(datetime.datetime.now()) + "/"

    util.setup(path)

    (image_ids, survival_days) = util.get_image_id_and_survival_days(exclude_pid = [186], glioma_grades=[4])

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
                                      image_registration.RIGID,
                                      save_to_db=True)
