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

    image_ids = range(455,465)
    util.LOGGER.info(str(image_ids) + " " + str(len(image_ids)))
    image_registration.get_transforms(image_ids,
                                      reg_type=image_registration.RIGID,
                                      save_to_db=True,
                                      reg_type_be=image_registration.SIMILARITY)

