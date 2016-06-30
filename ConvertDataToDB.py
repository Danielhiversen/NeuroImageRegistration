# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:19:49 2016

@author: dahoiv
"""
# pylint: disable= line-too-long
from __future__ import print_function

import errno
import glob
import os
import shutil
import sqlite3
import pyexcel_xlsx

import image_registration

# DATA_PATH_LISA = main_folder + "Segmenteringer_Lisa/"
# PID_LISA = main_folder + "Koblingsliste__Lisa.xlsx"
# DATA_PATH_LISA_QOL = main_folder + "Segmenteringer_Lisa/Med_QoL/"
# DATA_PATH_ANNE_LISE = main_folder + "Segmenteringer_AnneLine/"
# PID_ANNE_LISE = main_folder + "Koblingsliste__Anne_Line.xlsx"
# DATA_PATH_LGG = main_folder + "Data_HansKristian_LGG/LGG/NIFTI/"

main_folder = "/home/dahoiv/nevro_data/Segmentations/"
DWICONVERT_PATH = "/home/dahoiv/disk/kode/Slicer/Slicer-SuperBuild/Slicer-build/lib/Slicer-4.5/cli-modules/DWIConvert"


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
    `comments`    TEXT,
    FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')
    cursor.execute('''CREATE TABLE "Labels" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `image_id`    INTEGER NOT NULL,
    `description`    TEXT,
    `filepath`    TEXT,
    `comments`    TEXT,
    FOREIGN KEY(`image_id`) REFERENCES `Images`(`id`))''')
    cursor.execute('''CREATE TABLE "Patient" (
    `pid`    INTEGER NOT NULL UNIQUE,
    `comments`    TEXT,
    PRIMARY KEY(pid))''')
    cursor.execute('''CREATE TABLE "QualityOfLife" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    INTEGER NOT NULL,
    'Index_value'     REAL,
    'Global_index'    INTEGER,
    'Mobility'    INTEGER,
    'Selfcare'    INTEGER,
    'Activity'    INTEGER,
    'Pain'    INTEGER,
    'Anxiety'    INTEGER,
    FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')

    conn.commit()
    cursor.close()

    conn.close()


def mkdir_p(path):
    """Make new folder if not exits"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# ==============================================================================
#
# def get_convert_table(path):
#     """Open xls file and read new pid"""
#     xls_data = pyexcel_xlsx.get_data(path)
#     convert_table = {}
#     if 'Ark1' in xls_data:
#         data = xls_data['Ark1']  # lisa
#     else:
#         data = xls_data  # anne lise
#
#     for row in data:
#         if not row:
#             continue
#         pid = row[0]
#         case_id = row[1]
#         date = row[2]
#         convert_table[case_id] = [pid, date]
#     return convert_table
#
#
# def convert_lisa_data(path, qol):
#     """Convert data from lisa"""
#     # pylint: disable= too-many-locals
#     convert_table = get_convert_table(PID_LISA)
#
#     conn = sqlite3.connect(image_registration.DB_PATH)
#     cursor = conn.cursor()
#
#     for case_id in range(350):
#         data_path = path + str(case_id) + "/"
#         if not os.path.exists(data_path):
#             continue
#
#         print(data_path)
#         pid = "mnhr_" + str(convert_table[case_id][0])
#         date = convert_table[case_id][1]
#         volume_label = glob.glob(data_path + '/*label.nrrd')
#         if len(volume_label) == 0:
#             volume_label = glob.glob(data_path + '/*label_1.nrrd')
#         if len(volume_label) == 0:
#             volume_label = glob.glob(data_path + '/Segmentation/*label.nrrd')
#
#         if len(volume_label) > 1:
#             print("Warning!!\n\n More than one file with label found \n", volume_label)
#             continue
#         volume_label = volume_label[0]
#         volume = volume_label.replace("-label", "")
#         if not os.path.exists(volume):
#             volume = glob.glob(data_path + '*.nrrd')
#             volume.remove(volume_label)
#             if len(volume) > 1:
#                 print("Warning!!\n\n More than one file with volume found \n", volume)
#                 continue
#             volume = volume[0]
#
#         shutil.copy(volume_label, "volume_label.nrrd")
#         shutil.copy(volume, "volume.nrrd")
#         volume_label = "volume_label.nrrd"
#         volume = "volume.nrrd"
#
#         cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
#         exist = cursor.fetchone()
#         if exist is None:
#             cursor.execute('''INSERT INTO Patient(pid, diagnose) VALUES(?,?)''', (pid, 'HGG'))
#
#         cursor.execute('''INSERT INTO Surgery(pid, date) VALUES(?,?)''', (pid, date))
#         cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''', (pid, 'MR', 'pre'))
#         img_id = cursor.lastrowid
#         cursor.execute('''INSERT INTO Labels(image_id, segmented_by, description) VALUES(?,?,?)''', (img_id, 'Lisa', 'all'))
#         label_id = cursor.lastrowid
#         if qol:
#             cursor.execute('''INSERT INTO QualityOfLife(pid, qol) VALUES(?,?)''', (pid, -1))
#
#         mkdir_p(image_registration.DATA_FOLDER + str(pid))
#         img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/NIFTI/"
#         mkdir_p(img_out_folder)
#
#         volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_PRE.nii.gz"
#         volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_PRE_label_" + ".nii.gz"
#
#         volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
#         volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
#         cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))
#         cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''', (volume_label_out_db, label_id))
#
#         os.system(DWICONVERT_PATH + " --inputVolume " + volume + " -o " + volume_out + " --conversionMode NrrdToFSL")
#         os.system(DWICONVERT_PATH + " --inputVolume " + volume_label + " -o " + volume_label_out + " --conversionMode NrrdToFSL")
#         os.remove(volume)
#         os.remove(volume_label)
#
#         conn.commit()
#     cursor.close()
#     conn.close()
#
#
# def convert_annelise_data(path):
#     """Convert data from anne lise"""
#     # pylint: disable= too-many-locals
#     convert_table = get_convert_table(PID_ANNE_LISE)
#     conn = sqlite3.connect(image_registration.DB_PATH)
#     cursor = conn.cursor()
#
#     labels = {'hele-label': 'all', 'nekrose-label': 'nekrose', 'kontrast-label': 'kontrast'}
#     image_types = {"diag": "diag", "preop": "pre"}
#     ids = range(350)
#     ids.append("249b")
#     for case_id in ids:
#         volumes = glob.glob(path + "k" + str(case_id) + '_*.nii')
#
#         if len(volumes) == 0:
#             continue
#
#         if convert_table[case_id][0]:
#             pid = "mnhr_" + str(convert_table[case_id][0])
#         else:
#             pid = "annelise_" + str(case_id)
#
#         date = convert_table[case_id][1]
#         cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
#         exist = cursor.fetchone()
#         if exist is None:
#             cursor.execute('''INSERT INTO Patient(pid, diagnose) VALUES(?,?)''', (pid, 'HGG'))
#         cursor.execute('''INSERT INTO Surgery(pid, date) VALUES(?,?)''', (pid, date))
#         mkdir_p(image_registration.DATA_FOLDER + str(pid))
#         img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/NIFTI/"
#         mkdir_p(img_out_folder)
#
#         for image_type in image_types.keys():
#             volume = path + "k" + str(case_id) + "-T1_" + image_type + ".nii"
#             if not os.path.exists(volume):
#                 continue
#             cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
#                            (pid, 'MR', image_types[image_type]))
#             img_id = cursor.lastrowid
#
#             volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_types[image_type] + ".nii.gz"
#             os.system(DWICONVERT_PATH + " --inputVolume " + volume + " -o " + volume_out + " --conversionMode NrrdToFSL")
#             volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
#             cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))
#
#             for label_type in labels.keys():
#                 volume_label = path + "k" + str(case_id) + "_" + image_type + "_" + label_type + ".nii"
#                 if not os.path.exists(volume_label):
#                     continue
#
#                 cursor.execute('''INSERT INTO Labels(image_id, segmented_by, description) VALUES(?,?,?)''',
#                                (img_id, 'Anne Lise', labels[label_type]))
#                 label_id = cursor.lastrowid
#
#                 volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_types[image_type]\
#                     + "_label_" + labels[label_type] + ".nii.gz"
#                 os.system(DWICONVERT_PATH + " --inputVolume " + volume_label + " -o " +
#                           volume_label_out + " --conversionMode NrrdToFSL")
#                 volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
#                 cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''',
#                                (volume_label_out_db, label_id))
#
#         conn.commit()
#     cursor.close()
#     conn.close()
#
# ==============================================================================


def convert_and_save_dataset(pid, cursor, image_type, volume_labels, volume):
    mkdir_p(image_registration.DATA_FOLDER + str(pid))
    img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/volumes_labels/"
    mkdir_p(img_out_folder)

    cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
    exist = cursor.fetchone()
    if exist is None:
        cursor.execute('''INSERT INTO Patient(pid) VALUES(?)''', (pid, ))

    cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
                   (pid, 'MR', image_type))
    img_id = cursor.lastrowid

    filename, file_extension = os.path.splitext(volume)
    volume_temp = "volume" + file_extension
    shutil.copy(volume, volume_temp)

    volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type + ".nii.gz"
    print("--->", volume_out)
    os.system(DWICONVERT_PATH + " --inputVolume " + volume_temp + " -o " + volume_out + " --conversionMode NrrdToFSL")
    volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
    cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))
    os.remove(volume_temp)

    for volume_label in volume_labels:
        filename, file_extension = os.path.splitext(volume_label)
        volume_label_temp = "volume_label" + file_extension
        shutil.copy(volume_label, volume_label_temp)

        cursor.execute('''INSERT INTO Labels(image_id, description) VALUES(?,?)''',
                       (img_id, 'all'))
        label_id = cursor.lastrowid

        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type\
            + "_label_all.nii.gz"
        os.system(DWICONVERT_PATH + " --inputVolume " + volume_label_temp + " -o " +
                  volume_label_out + " --conversionMode NrrdToFSL")
        volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
        cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''',
                       (volume_label_out_db, label_id))
        os.remove(volume_label_temp)


def convert_gbm_data(path):
    """Convert gbm data """
    # pylint: disable= too-many-locals

    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.cursor()

    for case_id in range(2000):
        data_path = path + str(case_id) + "/"
        if not os.path.exists(data_path):
            continue

        pid = str(case_id)

        data_path = path + str(case_id) + "/"
        if not os.path.exists(data_path):
            continue

        print(data_path)

        file_type_nrrd = True
        volume_label = glob.glob(data_path + '/*label.nrrd')
        if len(volume_label) == 0:
            volume_label = glob.glob(data_path + '/*label_1.nrrd')
        if len(volume_label) == 0:
            volume_label = glob.glob(data_path + '/Segmentation/*label.nrrd')
        if len(volume_label) > 1:
            print("Warning!!\n\n More than one file with label found \n", volume_label)
            continue
        if len(volume_label) == 0:
            file_type_nrrd = False
            volume = data_path + "k" + pid + "-T1_" + "preop" + ".nii"
            if not os.path.exists(volume):
                print("Warning!!\n\n No volumes found \n", data_path, volume)

            volume_label = data_path + "k" + pid + "_" + "preop" + "_" + "hele-label" + ".nii"
            if not os.path.exists(volume_label):
                print("Warning!!\n\n No label found \n", data_path, volume_label)

        if file_type_nrrd:
            volume_label = volume_label[0]
            volume = volume_label.replace("-label", "")
            if not os.path.exists(volume):
                volume = glob.glob(data_path + '*.nrrd')
                volume.remove(volume_label)
                if len(volume) == 0:
                    volume = glob.glob(data_path + '*.nii')
                if len(volume) > 1:
                    print("Warning!!\n\n More than one file with volume found \n", volume)
                if len(volume) == 0:
                    print("Warning!!\n\n No volume found \n", data_path)
                volume = volume[0]

        convert_and_save_dataset(pid, cursor, "pre", [volume_label], volume)

        conn.commit()
    cursor.close()
    conn.close()


def qol_to_db():
    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.cursor()

    cursor.executescript('drop table if exists QualityOfLife;')
    cursor.execute('''CREATE TABLE "QualityOfLife" (
        `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        `pid`    INTEGER NOT NULL,
        'Index_value'     REAL,
        'Global_index'    INTEGER,
        'Mobility'    INTEGER,
        'Selfcare'    INTEGER,
        'Activity'    INTEGER,
        'Pain'    INTEGER,
        'Anxiety'    INTEGER,
        FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')

    data = pyexcel_xlsx.get_data('/home/dahoiv/Desktop/Indexverdier_atlas.xlsx')['Ark2']

    for row in data[3:]:
        print(row)
        if not row:
            continue

        if row[1] is None:
            gl_idx = None
        elif row[1] > 0.85:
            gl_idx = 1
        elif row[1] > 0.50:
            gl_idx = 2
        else:
            gl_idx = 3

        val = [None]*8
        idx = 0
        for _val in row[:8]:
            val[idx] = _val
            idx += 1
        cursor.execute('''INSERT INTO QualityOfLife(pid, Index_value, Global_index, Mobility, Selfcare, Activity, Pain, Anxiety) VALUES(?,?,?,?,?,?,?,?)''',
                       (val[0], val[1], gl_idx, val[2], val[3], val[4], val[5], val[6]))
        conn.commit()

    cursor.close()
    conn.close()


def convert_lgg_data(path):
    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.cursor()

    ids = range(350)
    for case_id in ids:
        volume = path + '%02d' % case_id + '_post.nii'
        image_type = 'post'
        if not os.path.exists(volume):
            volume = path + '%02d' % case_id + '_pre.nii'
            image_type = 'pre'
        if not os.path.exists(volume):
            continue
        print(volume, image_type)

        pid = str(case_id)

        if image_type == 'post':
            volume_label = path + '%02d' % case_id + '_post-label.nii'
        elif image_type == 'pre':
            volume_label = path + '%02d' % case_id + '_pre-label.nii'
        if not os.path.exists(volume):
            continue

        convert_and_save_dataset(pid, cursor, image_type, [volume_label], volume)
        conn.commit()

    cursor.close()
    conn.close()


def vacuum_db():
    """ Clean up database"""
    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.execute('''VACUUM; ''')
    cursor.close()
    conn.close()


if __name__ == "__main__":
    image_registration.setup_paths('GBM')
    try:
        shutil.rmtree(image_registration.DATA_FOLDER)
    except OSError:
        pass
    mkdir_p(image_registration.DATA_FOLDER)
    create_db(image_registration.DB_PATH)
    convert_gbm_data(main_folder + "Segmenteringer_GBM/")

    qol_to_db()
    vacuum_db()

    image_registration.setup_paths('LGG')
    try:
        shutil.rmtree(image_registration.DATA_FOLDER)
    except OSError:
        pass
    mkdir_p(image_registration.DATA_FOLDER)
    create_db(image_registration.DB_PATH)
    convert_lgg_data(main_folder + "Data_HansKristian_LGG/LGG/NIFTI/PRE_OP/")
    convert_lgg_data(main_folder + "Data_HansKristian_LGG/LGG/NIFTI/POST/")

    vacuum_db()
