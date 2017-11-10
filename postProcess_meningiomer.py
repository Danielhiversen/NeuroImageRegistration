# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import datetime
import nibabel as nib
import sqlite3

import util
from do_img_registration_meningiomer import find_images


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder, "meningiomer")

    image_ids = find_images(exclude=[1, 35, 192, 201, 463, 508, 530, 709, 866, 1020, 1061])

    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True, save_pngs=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


def find_bad_registration():
    util.setup(folder, "meningiomer")

    image_ids = find_images(exclude=[1, 35, 192, 201, 463, 508, 530, 709, 866, 1020, 1061])
    template_mask = nib.load(util.TEMPLATE_MASK).get_data()
    conn = sqlite3.connect(util.DB_PATH, timeout=120)
    k = 0
    for _id in image_ids:
        _filepath = conn.execute("SELECT filepath_reg from Labels where image_id = ?",
                                 (_id, )).fetchone()[0]
        com, com_idx = util.get_center_of_mass(util.DATA_FOLDER + _filepath)
        if template_mask[com_idx[0], com_idx[1], com_idx[2]] == 0:
            print(_filepath)
            k += 1
    print(k)


if __name__ == "__main__":
    folder = "RES_meningiomer_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"

    # process(folder)
    find_bad_registration()