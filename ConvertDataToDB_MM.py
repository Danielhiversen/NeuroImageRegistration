# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:19:49 2016

@author: dahoiv
"""
# pylint: disable= line-too-long
from __future__ import print_function

import glob
import os
import re
import shutil
import sqlite3
import pyexcel_xlsx

import util

# DATA_PATH_LISA = MAIN_FOLDER + "Segmenteringer_Lisa/"
# PID_LISA = MAIN_FOLDER + "Koblingsliste__Lisa.xlsx"
# DATA_PATH_LISA_QOL = MAIN_FOLDER + "Segmenteringer_Lisa/Med_QoL/"
# DATA_PATH_ANNE_LISE = MAIN_FOLDER + "Segmenteringer_AnneLine/"
# PID_ANNE_LISE = MAIN_FOLDER + "Koblingsliste__Anne_Line.xlsx"
# DATA_PATH_LGG = MAIN_FOLDER + "Data_HansKristian_LGG/LGG/NIFTI/"

MAIN_FOLDER = "/home/dahoiv/disk/data/MolekylareMarkorer_org/"
DWICONVERT_PATH = "/home/dahoiv/disk/kode/Slicer/Slicer-SuperBuild/Slicer-build/lib/Slicer-4.6/cli-modules/DWIConvert"


def create_db(path):
    """Make the database"""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE "Images" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    INTEGER,
    `modality`    TEXT,
    `diag_pre_post`    TEXT,
    `filepath`    TEXT,
    `transform`    TEXT,
    `fixed_image`    INTEGER,
    `filepath_reg`    TEXT,
    `comments`    TEXT,
    FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')
    cursor.execute('''CREATE TABLE "Labels" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `image_id`    INTEGER NOT NULL,
    `description`    TEXT,
    `filepath`    TEXT,
    `filepath_reg`    TEXT,
    `comments`    TEXT,
    FOREIGN KEY(`image_id`) REFERENCES `Images`(`id`))''')
    cursor.execute('''CREATE TABLE "Patient" (
    `pid`    INTEGER NOT NULL UNIQUE,
    `glioma_grade`    INTEGER,
    `comments`    TEXT,
    PRIMARY KEY(pid))''')
    cursor.execute('''CREATE TABLE "MolekylareMarkorer" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    INTEGER NOT NULL,
    'Subgroup'    INTEGER,
    `comments`    TEXT,
    FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')

    conn.commit()
    cursor.close()

    conn.close()


# pylint: disable= too-many-arguments, too-many-locals
def convert_and_save_dataset(pid, cursor, image_type, volume_labels, volume, glioma_grade, subgroup, comment):
    """convert_and_save_dataset"""
    util.mkdir_p(util.DATA_FOLDER + str(pid))
    img_out_folder = util.DATA_FOLDER + str(pid) + "/volumes_labels/"
    util.mkdir_p(img_out_folder)

    cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
    exist = cursor.fetchone()
    if exist is None:
        cursor.execute('''INSERT INTO Patient(pid, glioma_grade) VALUES(?, ?)''', (pid, glioma_grade))

    cursor.execute('''INSERT INTO MolekylareMarkorer(pid, Subgroup, comments) VALUES(?,?,?)''',
                   (pid, subgroup, comment))

    cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
                   (pid, 'MR', image_type))
    img_id = cursor.lastrowid

    _, file_extension = os.path.splitext(volume)
    volume_temp = "volume" + file_extension
    shutil.copy(volume, volume_temp)

    volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type + ".nii.gz"
    print("--->", volume_out)
    os.system(DWICONVERT_PATH + " --inputVolume " + volume_temp + " -o " + volume_out +
              " --conversionMode NrrdToFSL --allowLossyConversion")
    volume_out_db = volume_out.replace(util.DATA_FOLDER, "")
    cursor.execute('''UPDATE Images SET filepath = ?, filepath_reg = ? WHERE id = ?''', (volume_out_db, None, img_id))
    os.remove(volume_temp)

    for volume_label in volume_labels:
        _, file_extension = os.path.splitext(volume_label)
        volume_label_temp = "volume_label" + file_extension
        shutil.copy(volume_label, volume_label_temp)

        cursor.execute('''INSERT INTO Labels(image_id, description) VALUES(?,?)''',
                       (img_id, 'all'))
        label_id = cursor.lastrowid

        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type\
            + "_label_all.nii.gz"
        os.system(DWICONVERT_PATH + " --inputVolume " + volume_label_temp + " -o " +
                  volume_label_out + " --conversionMode NrrdToFSL --allowLossyConversion")
        volume_label_out_db = volume_label_out.replace(util.DATA_FOLDER, "")
        cursor.execute('''UPDATE Labels SET filepath = ?, filepath_reg = ? WHERE id = ?''',
                       (volume_label_out_db, None, label_id))
        os.remove(volume_label_temp)


def subgroup_to_db(data_type):
    # pylint: disable= too-many-branches
    """Convert qol data to database """
    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()

    data = pyexcel_xlsx.get_data('/mnt/dokumneter/data/Segmentations/Indexverdier_atlas_041116.xlsx')['Ark2']

    k = 0
    for row in data:
        k = k + 1
        if not row:
            continue
        if data_type == "extra" and k < 53:
            continue
        elif k < 4:
            continue

        if data_type == "gbm":
            idx1 = 1
            idx2 = 0
            idx3 = 7
        elif data_type == "lgg" or data_type == "extra":
            idx1 = 12
            idx2 = 11
            idx3 = 18
        if len(row) < idx3 - 1:
            continue
        print(row)

        if row[idx1] is None:
            gl_idx = None
        elif row[idx1] > 0.85:
            gl_idx = 1
        elif row[idx1] > 0.50:
            gl_idx = 2
        else:
            gl_idx = 3

        val = [None]*7
        idx = 0
        for _val in row[idx2:idx3]:
            val[idx] = _val
            idx += 1
        pid = val[0]
        cursor.execute('''SELECT id from QualityOfLife where pid = ?''', (pid,))
        _id = cursor.fetchone()
        if _id:
            print("-----------", row)
            continue
        cursor.execute('''INSERT INTO QualityOfLife(pid, Index_value, Global_index, Mobility, Selfcare, Activity, Pain, Anxiety) VALUES(?,?,?,?,?,?,?,?)''',
                       (pid, val[1], gl_idx, val[2], val[3], val[4], val[5], val[6]))
        conn.commit()

    cursor.close()
    conn.close()


def convert_lgg_data(path):
    """convert_lgg_data"""
    data = pyexcel_xlsx.get_data('/home/dahoiv/disk/data/MolekylareMarkorer_org/MolekylæreMarkører_AJS_281116.xlsx')

    convert_table = {}
    k = 0
    for row in data:
        k = k + 1
        if not row:
            continue
        elif k < 3:
            continue
        case_id = row[0]
        if case_id is None or not isinstance(case_id, int):
            continue
        subgroup = row[1]
        if subgroup is None or not isinstance(subgroup, int):
            continue
        comment = row[2]

        convert_table[case_id] = (subgroup, comment)

    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()
    for volume in glob.glob(path + "*.nrrd"):
        if "label" in volume:
            continue
        case_id = re.findall(r'\b\d+\b', volume)
        if len(case_id) != 1:
            print("ERROR", volume, case_id)
            return
        case_id = case_id[0]
        print(volume)
        if not os.path.exists(volume):
            continue

        image_type = 'pre'
        print(volume, image_type, case_id, comment)

        volume_label = path + str(case_id) + '-label.nrrd'
        if not os.path.exists(volume_label):
            continue

        convert_and_save_dataset(case_id, cursor, image_type, [volume_label], volume, 2, subgroup, comment)
        conn.commit()

    cursor.close()
    conn.close()


def vacuum_db():
    """ Clean up database"""
    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.execute('''VACUUM; ''')
    cursor.close()
    conn.close()


if __name__ == "__main__":
    util.setup_paths(data='MolekylareMarkorer')
    try:
        shutil.rmtree(util.DATA_FOLDER)
    except OSError:
        pass
    util.mkdir_p(util.DATA_FOLDER)
    create_db(util.DB_PATH)
    convert_lgg_data(MAIN_FOLDER)

    vacuum_db()
