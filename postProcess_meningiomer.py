# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
from os.path import basename
import nipype.interfaces.ants as ants
import datetime
import sqlite3

from img_data import img_data
import util
from do_img_registration_meningiomer import find_images


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder, "meningiomer")

    image_ids = find_images()

    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


if __name__ == "__main__":
    folder = "RES_meningiomer_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"

    process(folder)
