# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import datetime
import util


def process(folder):
    """ Post process data """
    print(folder)
    util.setup(folder)

    for grade in [2, 3, 4]:
        image_ids, survival_days = util.get_image_id_and_survival_days(glioma_grades=[grade])
        result = util.post_calculations(image_ids)

        print(len(result['all']))
        util.avg_calculation(result['all'], 'all_', None, True, folder, save_sum=True)
        util.avg_calculation(result['img'], 'img_', None, True, folder)

        for label in result:
            if label == 'img':
                continue
            util.avg_calculation(result[label], 'survival_time_grade_' + str(grade), survival_days, True, folder, default_value=-1)


if __name__ == "__main__":
    folder = "RES_survival_time_" + "{:%d%m%Y_%H%M}".format(datetime.datetime.now()) + "/"
    process(folder)
