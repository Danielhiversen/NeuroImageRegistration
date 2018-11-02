# pylint: disable= invalid-name
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:02 2016

@author: dahoiv
"""
from __future__ import print_function
import datetime
import os
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

    image_ids, survival_days = util.get_image_id_and_survival_days(study_id="GBM_survival_time", registration_date_upper_lim="2018-10-29")
    #image_ids = [10]
    #image_ids = [10, 19, 35, 371, 71, 83,98, 103, 106, 116, 231, 392, 458]
    #image_ids = [10, 19, 71, 83, 98, 103, 106, 116, 231, 392, 458]
    #image_ids = range(454,465)

    util.LOGGER.info(str(image_ids) + " " + str(len(image_ids)))
    image_registration.get_transforms(image_ids,
                                      #reg_type=image_registration.RIGID,
                                      #save_to_db=True,
                                      #reg_type_be=image_registration.SIMILARITY)
                                      reg_type=image_registration.SYN,
                                      reg_type_be=image_registration.SIMILARITY,
                                      save_to_db=True)
