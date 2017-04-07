# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import datetime
import sys

import util
from do_img_registration_GBM import find_images


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    params = ['Delta_qol2', 'Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety', 'karnofsky', 'Index_value', 'Delta_qol', 'Delta_kps']
    image_ids = find_images()
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    #util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    #util.avg_calculation(result['img'], 'img', None, True, folder)

    for qol_param in params:
        if qol_param == "Delta_qol2":
            (image_ids_with_qol, qol) = util.get_qol(image_ids, "Delta_qol")
            print(qol)
            qol = [-1 if _temp <= -0.15 else 0 if _temp < 0.15 else 1 for _temp in qol]
            print(qol)
        else:
            (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        if qol_param not in ["karnofsky", "Delta_kps"]:
            qol = [_temp * 100 for _temp in qol]
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
    params = ['Index_value', 'karnofsky',  'Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety']
    alternative = ['less', 'less', 'greater', 'greater', 'greater', 'greater', 'greater']
    stat_func = [util.brunner_munzel_test, util.mannwhitneyu_test, util.mannwhitneyu_test, util.mannwhitneyu_test,
                 util.mannwhitneyu_test, util.mannwhitneyu_test, util.mannwhitneyu_test]
    for (qol_param, stat_func_i, alternative_i) in zip(params, stat_func, alternative):
        (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            print(label)
            if label == 'img':
                continue
            util.vlsm(result[label], label + '_' + qol_param, stat_func_i, qol, folder,
                      n_permutations=n_permutations, alternative=alternative_i)


if __name__ == "__main__":
    folder = "RES_GBM_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"
    process(folder)
    start_time = datetime.datetime.now()
    if len(sys.argv) > 1:
        n_permutations = int(sys.argv[1])
    else:
        n_permutations = 20
    # process_vlsm(folder, n_permutations)
    print("Total runtime")
    print(datetime.datetime.now() - start_time)
