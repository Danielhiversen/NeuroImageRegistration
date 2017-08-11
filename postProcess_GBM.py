# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import datetime
# import sys
import sqlite3

import util
import do_img_registration_GBM


def find_images():
    """ Find images for registration """
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient where study_id = ?''', ("qol_grade3,4", ))
    ids = []
    for row in cursor:
        cursor3 = conn.execute('''SELECT Resection from QualityOfLife where pid = ?''', (row[0], ))
        resection = cursor3.fetchone()[0]
        cursor3.close()
        if resection in [0, None]:
            continue
        cursor2 = conn.execute('''SELECT id from Images where pid = ?''', (row[0], ))
        for _id in cursor2:
            ids.append(_id[0])
        cursor2.close()

    cursor.close()
    conn.close()
    return ids


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    params = ['Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety', 'karnofsky', 'Index_value']
    image_ids = do_img_registration_GBM.find_images()
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)

    for qol_param in params:
        if qol_param == "Delta_qol2":
            (image_ids_with_qol, qol) = util.get_qol(image_ids, "Delta_qol")
            qol = [-1 if _temp <= -0.15 else 0 if _temp < 0.15 else 1 for _temp in qol]
        else:
            (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        if qol_param not in ["karnofsky", "Delta_kps"]:
            qol = [_temp * 100 for _temp in qol]
        default_value = -100
        print(qol_param)
        print(len(qol))
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            if label == 'img':
                continue
            print(label)
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, folder, default_value=default_value)
            util.median_calculation(result[label], label + '_' + qol_param, qol, True, folder, default_value=default_value)
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


def process2(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    params = ['Delta_qol', 'Delta_qol2', 'Delta_mobility', 'Delta_selfcare', 'Delta_activity', 'Delta_pain', 'Delta_anixety', 'Delta_kps']
    image_ids = find_images()
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)
    print("\n\n\n\n\n")

    for qol_param in params:
        if qol_param == "Delta_qol2":
            (image_ids_with_qol, qol) = util.get_qol(image_ids, "Delta_qol")
            qol = [-1 if _temp <= -0.15 else 0 if _temp < 0.15 else 1 for _temp in qol]
        else:
            (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        qol = [_temp * 100 for _temp in qol]
        default_value = -200
        print(qol_param, len(qol))
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            if label == 'img':
                continue
            print(label)
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, folder, default_value=default_value)
            util.median_calculation(result[label], label + '_' + qol_param, qol, True, folder, default_value=default_value)
            # util.std_calculation(result[label], label + '_' + qol_param, qol, True, folder)


def process3(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)
    params = ['Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety', 'karnofsky', 'Index_value']
    image_ids = find_images()
    result = util.post_calculations(image_ids)
    print(len(result['all']))
    util.avg_calculation(result['all'], 'all', None, True, folder, save_sum=True)
    util.avg_calculation(result['img'], 'img', None, True, folder)

    for qol_param in params:
        if qol_param == "Delta_qol2":
            (image_ids_with_qol, qol) = util.get_qol(image_ids, "Delta_qol")
            qol = [-1 if _temp <= -0.15 else 0 if _temp < 0.15 else 1 for _temp in qol]
        else:
            (image_ids_with_qol, qol) = util.get_qol(image_ids, qol_param)
        if qol_param not in ["karnofsky", "Delta_kps"]:
            qol = [_temp * 100 for _temp in qol]
        default_value = -100
        print(qol_param)
        print(len(qol))
        result = util.post_calculations(image_ids_with_qol)
        for label in result:
            if label == 'img':
                continue
            print(label)
            util.avg_calculation(result[label], label + '_' + qol_param + '_N=112', qol, True, folder, default_value=default_value)
            util.median_calculation(result[label], label + '_' + qol_param + '_N=112', qol, True, folder, default_value=default_value)
            # util.std_calculation(result[label], label + '_' + qol_param, qol, True, folder)


if __name__ == "__main__":
    folder = "RES_GBM_" + "{:%H%M_%m_%d_%Y}".format(datetime.datetime.now()) + "/"
    process3(folder)
    process(folder)
    process2(folder)

    # start_time = datetime.datetime.now()
    # if len(sys.argv) > 1:
    #     n_permutations = int(sys.argv[1])
    # else:
    #     n_permutations = 20
    # # process_vlsm(folder, n_permutations)
    # print("Total runtime")
    # print(datetime.datetime.now() - start_time)
