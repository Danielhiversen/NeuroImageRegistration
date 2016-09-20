# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:34:30 2016

@author: dahoiv
"""

import os
import sqlite3

import ConvertDataToDB
from img_data import img_data
import image_registration
import util

if __name__ == "__main__":
    os.nice(19)

    util.setup("temp_convert/", "LGG")
    util.mkdir_p(util.TEMP_FOLDER_PATH)

    util.DATA_FOLDER = "/mnt/dokumneter/data/database3/"

    data_transforms = []

    if True:
        db_path = "/home/dahoiv/disk/data/database/LGG/"
        util.DATA_FOLDER = util.DATA_FOLDER + "LGG" + "/"
        util.DB_PATH = util.DATA_FOLDER + "brainSegmentation.db"

        convert_table_inv = ConvertDataToDB.get_convert_table('/home/dahoiv/disk/data/Segmentations/NY_PID_LGG segmentert.xlsx')
        convert_table = {v: k for k, v in convert_table_inv.items()}
        print(convert_table_inv)

        conn = sqlite3.connect(util.DB_PATH)
        conn.text_factory = str
        cursor = conn.execute('''SELECT pid from Patient''')

        conn2 = sqlite3.connect(db_path + "brainSegmentation.db")
        conn2.text_factory = str
        image_ids = []
        ny_image_ids = []
        for row in cursor:
            print(row)
            ny_pid = row[0]
            try:
                old_pid = int(convert_table[str(ny_pid)])
            except:
                continue
            cursor2 = conn2.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''', (old_pid, "pre"))
            for _id in cursor2:
                image_ids.append(_id[0])
            cursor2.close()

            cursor2 = conn.execute('''SELECT id from Images where pid = ? AND diag_pre_post = ?''', (ny_pid, "pre"))
            for _id in cursor2:
                ny_image_ids.append(_id[0])
            cursor2.close()

        cursor.close()
        conn.close()
    if True:
        db_path = "/home/dahoiv/disk/data/database/GBM/"

        util.DATA_FOLDER = util.DATA_FOLDER + "GBM" + "/"
        util.DB_PATH = util.DATA_FOLDER + "brainSegmentation.db"

        import do_img_registration_GBM
        image_ids = do_img_registration_GBM.find_images()
        ny_image_ids = image_ids

    for (img_id, ny_img_id) in zip(image_ids, ny_image_ids):
        print(img_id)
        _img = img_data(img_id, db_path, util.TEMP_FOLDER_PATH)
        _img.load_db_transforms()
        if _img.transform is None:
            continue
        _img.processed_filepath = image_registration.move_vol(_img.img_filepath, _img.get_transforms())
        _img.image_id = ny_img_id
        data_transforms.append(_img)

    image_registration.save_transform_to_database(data_transforms)
