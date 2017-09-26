# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import datetime
import util
from do_img_registration_meningiomer import find_images


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder, "meningiomer")

    image_ids = find_images()

    result = util.post_calculations(image_ids)
    print(result)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True, save_pngs=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)


if __name__ == "__main__":
    folder = "RES_meningiomer_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"

    process(folder)
