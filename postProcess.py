# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import os
import util

if __name__ == "__main__":
    os.nice(19)
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
    util.mkdir_p("LGG_GBM_RES/LGG")
    util.mkdir_p("LGG_GBM_RES/GBM")

    for qol_param in params:
        util.setup("LGG_POST_RES/LGG/", "LGG")
        util.mkdir_p(util.TEMP_FOLDER_PATH)
        (image_ids, qol) = util.get_image_id_and_qol(qol_param, True)
        print(image_ids)
        result = util.post_calculations(image_ids)
        for label in result:
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, "LGG_GBM_RES")

        print(result)

        util.setup("GBM_RES/GBM/", "GBM")
        util.mkdir_p(util.TEMP_FOLDER_PATH)
        (image_ids, _qol) = util.get_image_id_and_qol(qol_param)
        qol.extend(_qol)
        result = util.post_calculations(image_ids, result)
        print(result)

        for label in result:
            util.avg_calculation(result[label], label + '_' + qol_param, qol, True, "LGG_GBM_RES")
