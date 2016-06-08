# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:17:43 2016

@author: dahoiv
"""


import os

import ConvertDataToDB
import image_registration
import do_img_registration_HGG as do_img_registration

if __name__ == "__main__":
    os.nice(19)
    image_registration.setup("HGG/")
    image_registration.DATA_FOLDER = "/home/dahoiv/database_hgg/"
    image_registration.DB_PATH = image_registration.DATA_FOLDER + "brainSegmentation.db"

    ConvertDataToDB.copy_transforms(do_img_registration.find_images(), "/mnt/dokumenter/daniel/database/")
