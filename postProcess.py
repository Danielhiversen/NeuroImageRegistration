# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

# import os
import util

if __name__ == "__main__":
    #    if False:
    #        import do_img_registration_LGG_POST as do_img_registration
    #        util.setup("LGG_POST_RES/", "LGG")
    #    elif False:
    #        import do_img_registration_LGG_PRE as do_img_registration
    #        util.setup("LGG_PRE_RES/", "LGG")
    #    elif False:
    #        import do_img_registration_GBM as do_img_registration
    #        util.setup("GBM_RES2/", "GBM")

    params = ['Index_value', 'Global_index', 'Mobility', 'Selfcare', 'Activity', 'Pain', 'Anxiety']
    util.mkdir_p("LGG_GBM_RES")

    FOLDER = "LGG_GBM_RES3/"  # "LGG_GBM_RES/GBM"
    util.setup(FOLDER)

    image_ids = util.find_images_with_qol()
    result = util.post_calculations(image_ids)
    util.avg_calculation(result['img'], 'img', None, True, FOLDER)
    util.avg_calculation(result['all'], 'all', None, True, FOLDER)
    util.sum_calculation(result['all'], 'all', None, True, FOLDER)

    (image_ids, qol) = util.get_image_id_and_qol('Index_value')
    print(image_ids, len(image_ids))
    result = util.post_calculations(image_ids)
    util.calculate_t_test(result['all'], 1)

    for qol_param in params:
        (image_ids, qol) = util.get_image_id_and_qol(qol_param)
        if qol_param == 'Index_value':
            qol = [(_temp+1) * 100 for _temp in qol]
        else:
            qol = [_temp * 100 for _temp in qol]
        print(image_ids)
        result = util.post_calculations(image_ids)
        for label in result:
            print(label)
            if label == 'img':
                continue
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, FOLDER)
