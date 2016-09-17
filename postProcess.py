# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import os
import util

if __name__ == "__main__":
    os.nice(19)
    if False:
        import do_img_registration_LGG_POST as do_img_registration
        util.setup("LGG_POST_RES/", "LGG")
    elif False:
        import do_img_registration_LGG_PRE as do_img_registration
        util.setup("LGG_PRE_RES/", "LGG")
    elif False:
        import do_img_registration_GBM as do_img_registration
        util.setup("GBM_RES2/", "GBM")


    qol_param = 'Index_value'

    util.setup("LGG_POST_RES/", "LGG")
    if not os.path.exists(util.TEMP_FOLDER_PATH):
        os.makedirs(util.TEMP_FOLDER_PATH)
    (image_ids, qol) = util.get_image_id_and_qol(qol_param)
    result = util.post_calculations(image_ids)

    util.setup("GBM_RES2/", "GBM")
    if not os.path.exists(util.TEMP_FOLDER_PATH):
        os.makedirs(util.TEMP_FOLDER_PATH)
    (image_ids, _qol) = util.get_image_id_and_qol(qol_param)
    qol.extend(_qol)
    result.extend(util.post_calculations(image_ids))

    for label in result:
        util.avg_calculation(result[label], label, qol, True)


