# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:41:50 2016

@author: dahoiv
"""

import os
import image_registration

if __name__ == "__main__":
    os.nice(19)
    if True:
        import do_img_registration_LGG_POST as do_img_registration
        image_registration.setup("LGG_POST_RES/")

    if not os.path.exists(image_registration.TEMP_FOLDER_PATH):
        os.makedirs(image_registration.TEMP_FOLDER_PATH)

    image_ids = do_img_registration.find_images()
    image_registration.post_calculations(image_ids)
