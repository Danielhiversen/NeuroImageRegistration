# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import sys

import util
from do_img_registration_GBM import find_images

def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    params = ['Index_value', 'Global_index', 'Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety', 'karnofsky']
    image_ids = find_images()
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)

    for qol_param in params:
        (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        if not qol_param == "karnofsky":
            qol = [_temp * 100 for _temp in qol]
        if qol_param == "Index_value":
            default_value = 0
        else:
            default_value = -100
        print(qol_param)
        print(image_ids_with_qol)
        print(len(qol))
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            if label == 'img':
                continue
            print(label)
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, folder, default_value=default_value)
            # util.std_calculation(result[label], label + '_' + qol_param, qol, True, folder)


def process_vlsm(folder, n_permutations):
    """ Post process vlsm data """
    print(folder)
    util.setup(folder)
    image_ids = find_images()
    params = ['Index_value', 'Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety', 'karnofsky']
    stat_func = [util.brunner_munzel_test, util.mannwhitneyu_test, util.mannwhitneyu_test, util.mannwhitneyu_test,
                 util.mannwhitneyu_test, util.mannwhitneyu_test, util.mannwhitneyu_test]
    print(stat_func)
    for (qol_param, stat_func_i) in zip(params, stat_func):
        (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            print(label)
            if label == 'img':
                continue
            util.vlsm(result[label], label + '_' + qol_param, stat_func_i, qol, folder, n_permutations=n_permutations)


if __name__ == "__main__":
    import datetime

    folder = "RES/"
    process(folder)
    start_time = datetime.datetime.now()
    if len(sys.argv) > 2:
        n_permutations = int(sys.argv[1])
    else:
        n_permutations = 8

    process_vlsm(folder, n_permutations)
    print("Total runtime")
    print(datetime.datetime.now() - start_time)
