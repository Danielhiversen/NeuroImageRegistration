# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:00:00 2019

@author: leb
"""
import os
import shutil
import datetime
import sqlite3
import glob
import util
from img_data import img_data
from image_registration import move_vol






# pylint: disable= invalid-name
if __name__ == "__main__":  # if 'unity' in hostname or 'compute' in hostname:

#    new_segmentations_folder = '/Volumes/Neuro/Segmentations/oppdaterte_filer/'
    new_segmentations_folder = '/media/leb/data/oppdaterte_filer/'

    temp_folder = "ADD_SEGMENTATIONS_" + "{:%Y%m%d_%H%M}".format(datetime.datetime.now()) + "/"
    util.setup(temp_folder,'glioma')

    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()

    modified_patients = [subfolder_name for subfolder_name in os.listdir(new_segmentations_folder)
                         if os.path.isdir(os.path.join(new_segmentations_folder, subfolder_name))]

    for pid in modified_patients:
	#print('PATIENT ' + pid)
        cursor.execute("SELECT id FROM Images WHERE pid = ?", (int(pid),))
        image_id = cursor.fetchone()[0]
        print(image_id)
        img = img_data(image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img.load_db_transforms()
        #cursor.execute("SELECT  filepath, filepath_reg FROM Labels WHERE image_id IN "
        #               "(SELECT id FROM Images WHERE pid = ?)", (int(pid),))
        #(label_path, label_reg_path) = cursor.fetchone()
        #cursor.execute("SELECT  transform FROM Images WHERE pid = ?", (int(pid),))
        #transform = cursor.fetchone()[0]
        new_segmentation_path = glob.glob(new_segmentations_folder + str(pid)+'/*label.nii')[0]
        #print(new_segmentation_path)

        #temp = util.compress_vol(move_vol(new_segmentation_path, img.get_transforms(), True))
        #shutil.copy(temp, img.reg_label_filepath)
        shutil.copy(new_segmentation_path, img.label_filepath)


    #path = '/Volumes/Neuro/Segmentations/oppdaterte_filer/**/*label.nii'
    #glob.glob(path)

