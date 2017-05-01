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

MAIN_FOLDER = "/home/dahoiv/disk/data/MolekylareMarkorer/JAMA_tromso/"
# MAIN_FOLDER = "/home/dahoiv/disk/data/MolekylareMarkorer/MolekylareMarkorer_org/"
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
def convert_and_save_dataset(pid, cursor, image_type, volume_labels, volume, glioma_grade, subgroup, comment, tromso=False):
    """convert_and_save_dataset"""
    util.mkdir_p(util.DATA_FOLDER + str(pid))
    img_out_folder = util.DATA_FOLDER + str(pid) + "/volumes_labels/"
    util.mkdir_p(img_out_folder)

    cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
    exist = cursor.fetchone()

    patient_comment = ""
    if tromso:
        patient_comment = "Tromso"
    if exist is None:
        cursor.execute('''INSERT INTO Patient(pid, glioma_grade, comments) VALUES(?, ?, ?)''', (pid, glioma_grade, patient_comment))

    cursor.execute('''INSERT INTO MolekylareMarkorer(pid, Subgroup, comments) VALUES(?,?,?)''',
                   (pid, subgroup, comment))

    cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
                   (pid, 'MR', image_type))
    img_id = cursor.lastrowid

    volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type + ".nii.gz"

    if volume[-7:] == '.nii.gz':
        shutil.copy(volume, volume_out)
    else:
        _, file_extension = os.path.splitext(volume)
        volume_temp = "volume" + file_extension
        shutil.copy(volume, volume_temp)

        print("--->", volume_out)
        os.system(DWICONVERT_PATH + " --inputVolume " + volume_temp + " -o " + volume_out +
                  " --conversionMode NrrdToFSL --allowLossyConversion")
        os.remove(volume_temp)

    volume_out_db = volume_out.replace(util.DATA_FOLDER, "")
    cursor.execute('''UPDATE Images SET filepath = ?, filepath_reg = ? WHERE id = ?''', (volume_out_db, None, img_id))

    for volume_label in volume_labels:

        cursor.execute('''INSERT INTO Labels(image_id, description) VALUES(?,?)''',
                       (img_id, 'all'))
        label_id = cursor.lastrowid

        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type\
            + "_label_all.nii.gz"

        if volume[-7:] == '.nii.gz':
            shutil.copy(volume_label, volume_label_out)
        else:
            _, file_extension = os.path.splitext(volume_label)
            volume_label_temp = "volume_label" + file_extension
            shutil.copy(volume_label, volume_label_temp)

            os.system(DWICONVERT_PATH + " --inputVolume " + volume_label_temp + " -o " +
                      volume_label_out + " --conversionMode NrrdToFSL --allowLossyConversion")
            os.remove(volume_label_temp)

        volume_label_out_db = volume_label_out.replace(util.DATA_FOLDER, "")
        cursor.execute('''UPDATE Labels SET filepath = ?, filepath_reg = ? WHERE id = ?''',
                       (volume_label_out_db, None, label_id))


def convert_lgg_data(path):
    """convert_lgg_data"""
    data = pyexcel_xlsx.get_data('/home/dahoiv/disk/data/MolekylareMarkorer/MolekylareMarkorer_org/MolekylæreMarkører_AJS_281116.xlsx')

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
        case_id = int(case_id[0])
        print(volume)
        if not os.path.exists(volume):
            continue

        image_type = 'pre'
        print(volume, image_type, case_id, comment)

        volume_label = path + str(case_id) + '-label.nrrd'
        if not os.path.exists(volume_label):
            continue

        (subgroup, comment) = convert_table.get(case_id, (None, None))
        convert_and_save_dataset(case_id, cursor, image_type, [volume_label], volume, 2, subgroup, comment)
        conn.commit()

    cursor.close()
    conn.close()


def convert_lgg_data_tromso(path):
    """convert_lgg_data"""
    data = pyexcel_xlsx.get_data('/home/dahoiv/disk/data/MolekylareMarkorer/JAMA_tromso/MolekylareMarkorer_Tromso_AJS_04.01.2017.xlsx')

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
    for volume in glob.glob(path + "*.nii"):
        if "label" in volume:
            print(volume)
            continue
        case_id = re.findall(r'\d+\b', volume)
        if len(case_id) != 1:
            print("ERROR", volume, case_id)
            return
        case_id = int(case_id[0])
        print(volume)
        if not os.path.exists(volume):
            print("ERROR, volume missing", volume, case_id)
            return

        image_type = 'pre'
        print(volume, image_type, case_id, comment)

        volume_label = path + 'T' + str(case_id) + '-label.nii'
        if not os.path.exists(volume_label):
            volume_label = path + 'T' + str(case_id) + '_label.nii'
            if not os.path.exists(volume_label):
                print("ERROR, no label", volume_label, case_id)
                return

        (subgroup, comment) = convert_table.get(case_id, (None, None))
        convert_and_save_dataset(case_id, cursor, image_type, [volume_label], volume, 2, subgroup, comment, True)
        conn.commit()

    cursor.close()
    conn.close()


def convert_lgg_data_tromso_reseksjon(path):
    """convert_lgg_data"""
    convert_table = get_convert_table()

    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()
    for volume in glob.glob(path + "*.nii"):
        if "label" in volume:
            continue
        case_id = re.findall(r'\d+\b', volume)
        if len(case_id) != 1:
            print("ERROR", volume, case_id)
            return
        case_id = int(case_id[0])
        print(volume)
        if not os.path.exists(volume):
            print("ERROR, volume missing", volume, case_id)
            return

        image_type = 'pre'

        volume_label = path + 'T' + str(case_id) + '-label.nii'
        if not os.path.exists(volume_label):
            volume_label = path + 'T' + str(case_id) + '_label.nii'
            if not os.path.exists(volume_label):
                print("ERROR, no label", volume_label, case_id)
                return
        subgroup = convert_table.get(case_id, None)
        print(volume, image_type, case_id, subgroup)
        convert_and_save_dataset(case_id, cursor, image_type, [volume_label], volume, 2, subgroup, "", True)
        conn.commit()

    cursor.close()
    conn.close()


def get_convert_table():
    data = pyexcel_xlsx.get_data('/home/dahoiv/disk/data/MolekylareMarkorer/patientlist_norway.xlsx')['Ark1']
    convert_table = {}
    k = 0
    for row in data:
        k = k + 1
        if not row:
            continue
        pid = row[0]
        try:
            pid = int(pid)
        except ValueError:
            continue
        subgroup = row[3]
        try:
            subgroup = int(subgroup)
        except ValueError:
            continue
        convert_table[pid] = subgroup
    return convert_table


def add_from_gbm_db(path):
    """convert_lgg_data"""
    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()

    opids = [515, 527, 579, 600, 727, 728, 826, 840, 847, 857, 916, 934, 976, 980, 1030, 1070, 1084, 1124, 1176, 1195, 1197, 1211, 1166, 966, 1254, 1258, 1261, 1269, 1271, 1278, 1352, 1461, 1585, 1553, 1505, 1483, 1481, 666, 1454, 1432, 1219, 1297, 1307, 265, 408, 611, 678, 800, 805, 1146, 1189]  # noqa: E501

    convert_table = get_convert_table()

    data = pyexcel_xlsx.get_data('/home/dahoiv/disk/data/MolekylareMarkorer/pas_til_kart_oversikt.xlsx')['Ark1']
    convert_table_opid_to_pid = {}
    k = 0
    for row in data:
        k = k + 1
        if not row or len(row) < 13:
            continue
        opid = row[11]
        try:
            opid = int(opid)
        except ValueError:
            continue
        pid = row[12]
        try:
            pid = int(pid)
        except ValueError:
            continue
        convert_table_opid_to_pid[opid] = pid
    for opid in opids:
        volume = ""
        for _volume in glob.glob(path + str(opid) + "/volumes_labels/*.nii.gz"):
            if "_label_" not in _volume and "_pre" in _volume:
                volume = _volume
                break
        image_type = 'pre'
        volume_label = ""
        for _volume_label in glob.glob(path + str(opid) + "/volumes_labels/*.nii.gz"):
            if "_label_" in _volume_label and "_pre" in _volume_label:
                volume_label = _volume_label
                break
        pid = convert_table_opid_to_pid[opid]
        subgroup = convert_table.get(pid, None)
        print(pid, opid, subgroup, volume, volume_label)
        convert_and_save_dataset(pid, cursor, image_type, [volume_label], volume, 2, subgroup, "", True)
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
#    try:
#        shutil.rmtree(util.DATA_FOLDER)
#    except OSError:
#        pass
#    util.mkdir_p(util.DATA_FOLDER)
#    create_db(util.DB_PATH)
#     convert_lgg_data_tromso(MAIN_FOLDER)

    # convert_lgg_data_tromso_reseksjon("/home/dahoiv/disk/data/MolekylareMarkorer/JAMA_Tromso_reseksjon/")

    add_from_gbm_db("/home/dahoiv/disk/data/Segmentations/database/")

    vacuum_db()
