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
    ConvertDataToDB.copy_transforms(do_img_registration.find_images(), "/home/daniel/db_data/mnt/dokumenter/daniel/database/")