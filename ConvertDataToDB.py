# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:19:49 2016

@author: dahoiv
"""
# pylint: disable= line-too-long
from __future__ import print_function
from os.path import basename

import errno
import glob
import os
import pyexcel_xlsx
import shutil
import sqlite3

import image_registration


def create_db(path):
    """Make the database"""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE "Images" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    TEXT,
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
    `segmented_by`    TEXT,
    `comments`    TEXT,
    FOREIGN KEY(`image_id`) REFERENCES `Images`(`id`))''')
    cursor.execute('''CREATE TABLE "Patient" (
    `pid`    TEXT NOT NULL UNIQUE,
    `diagnose`    TEXT,
    `comments`    TEXT,
    PRIMARY KEY(pid))''')
    cursor.execute('''CREATE TABLE "QualityOfLife" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    TEXT NOT NULL,
    `qol`    INTEGER NOT NULL,
    `time`    INTEGER,
    FOREIGN KEY(`pid`) REFERENCES `Patient`(`pid`))''')
    cursor.execute('''CREATE TABLE "Surgery" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    TEXT NOT NULL,
    `date`    INTEGER,
    `comments`    TEXT,
    FOREIGN KEY(`pid`) REFERENCES Patient ( pid ))''')

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


def get_convert_table(path):
    """Open xls file and read new pid"""
    xls_data = pyexcel_xlsx.get_data(path)
    convert_table = {}
    if 'Ark1' in xls_data:
        data = xls_data['Ark1']  # lisa
    else:
        data = xls_data  # anne lise

    for row in data:
        if not row:
            continue
        pid = row[0]
        case_id = row[1]
        date = row[2]
        convert_table[case_id] = [pid, date]
    return convert_table


def convert_lisa_data(path, qol):
    """Convert data from lisa"""
    # pylint: disable= too-many-locals
    convert_table = get_convert_table(image_registration.PID_LISA)

    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.cursor()

    for case_id in range(350):
        data_path = path + str(case_id) + "/"
        if not os.path.exists(data_path):
            continue

        print(data_path)
        pid = "mnhr_" + str(convert_table[case_id][0])
        date = convert_table[case_id][1]
        volume_label = glob.glob(data_path + '/*label.nrrd')
        if len(volume_label) == 0:
            volume_label = glob.glob(data_path + '/*label_1.nrrd')
        if len(volume_label) == 0:
            volume_label = glob.glob(data_path + '/Segmentation/*label.nrrd')

        if len(volume_label) > 1:
            print("Warning!!\n\n More than one file with label found \n", volume_label)
            continue
        volume_label = volume_label[0]
        volume = volume_label.replace("-label", "")
        if not os.path.exists(volume):
            volume = glob.glob(data_path + '*.nrrd')
            volume.remove(volume_label)
            if len(volume) > 1:
                print("Warning!!\n\n More than one file with volume found \n", volume)
                continue
            volume = volume[0]

        shutil.copy(volume_label, "volume_label.nrrd")
        shutil.copy(volume, "volume.nrrd")
        volume_label = "volume_label.nrrd"
        volume = "volume.nrrd"

        cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
        exist = cursor.fetchone()
        if exist is None:
            cursor.execute('''INSERT INTO Patient(pid, diagnose) VALUES(?,?)''', (pid, 'HGG'))

        cursor.execute('''INSERT INTO Surgery(pid, date) VALUES(?,?)''', (pid, date))
        cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''', (pid, 'MR', 'pre'))
        img_id = cursor.lastrowid
        cursor.execute('''INSERT INTO Labels(image_id, segmented_by, description) VALUES(?,?,?)''', (img_id, 'Lisa', 'all'))
        label_id = cursor.lastrowid
        if qol:
            cursor.execute('''INSERT INTO QualityOfLife(pid, qol) VALUES(?,?)''', (pid, -1))

        mkdir_p(image_registration.DATA_FOLDER + str(pid))
        img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/NIFTI/"
        mkdir_p(img_out_folder)

        volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_PRE.nii.gz"
        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_PRE_label_" + ".nii.gz"

        volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
        volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
        cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))
        cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''', (volume_label_out_db, label_id))

        os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume + " -o " + volume_out + " --conversionMode NrrdToFSL")
        os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume_label + " -o " + volume_label_out + " --conversionMode NrrdToFSL")
        os.remove(volume)
        os.remove(volume_label)

        conn.commit()
    cursor.close()
    conn.close()


def convert_annelise_data(path):
    """Convert data from anne lise"""
    # pylint: disable= too-many-locals
    convert_table = get_convert_table(image_registration.PID_ANNE_LISE)
    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.cursor()

    labels = {'hele-label': 'all', 'nekrose-label': 'nekrose', 'kontrast-label': 'kontrast'}
    image_types = {"diag": "diag", "preop": "pre"}
    ids = range(350)
    ids.append("249b")
    for case_id in ids:
        volumes = glob.glob(path + "k" + str(case_id) + '_*.nii')

        if len(volumes) == 0:
            continue

        if convert_table[case_id][0]:
            pid = "mnhr_" + str(convert_table[case_id][0])
        else:
            pid = "annelise_" + str(case_id)

        date = convert_table[case_id][1]
        cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
        exist = cursor.fetchone()
        if exist is None:
            cursor.execute('''INSERT INTO Patient(pid, diagnose) VALUES(?,?)''', (pid, 'HGG'))
        cursor.execute('''INSERT INTO Surgery(pid, date) VALUES(?,?)''', (pid, date))
        mkdir_p(image_registration.DATA_FOLDER + str(pid))
        img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/NIFTI/"
        mkdir_p(img_out_folder)

        for image_type in image_types.keys():
            volume = path + "k" + str(case_id) + "-T1_" + image_type + ".nii"
            if not os.path.exists(volume):
                continue
            cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
                           (pid, 'MR', image_types[image_type]))
            img_id = cursor.lastrowid

            volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_types[image_type] + ".nii.gz"
            os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume + " -o " + volume_out + " --conversionMode NrrdToFSL")
            volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
            cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))

            for label_type in labels.keys():
                volume_label = path + "k" + str(case_id) + "_" + image_type + "_" + label_type + ".nii"
                if not os.path.exists(volume_label):
                    continue

                cursor.execute('''INSERT INTO Labels(image_id, segmented_by, description) VALUES(?,?,?)''',
                               (img_id, 'Anne Lise', labels[label_type]))
                label_id = cursor.lastrowid

                volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_types[image_type]\
                    + "_label_" + labels[label_type] + ".nii.gz"
                os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume_label + " -o " +
                          volume_label_out + " --conversionMode NrrdToFSL")
                volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
                cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''',
                               (volume_label_out_db, label_id))

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

        pid = "lgg_" + str(case_id)

        cursor.execute('''SELECT pid from Patient where pid = ?''', (pid,))
        exist = cursor.fetchone()
        if exist is None:
            cursor.execute('''INSERT INTO Patient(pid, diagnose) VALUES(?,?)''', (pid, 'LGG'))

        mkdir_p(image_registration.DATA_FOLDER + str(pid))
        img_out_folder = image_registration.DATA_FOLDER + str(pid) + "/NIFTI/"
        mkdir_p(img_out_folder)

        cursor.execute('''INSERT INTO Images(pid, modality, diag_pre_post) VALUES(?,?,?)''',
                       (pid, 'MR', image_type))
        img_id = cursor.lastrowid

        volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type + ".nii.gz"
        print("--->", volume_out)
        os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume + " -o " + volume_out + " --conversionMode NrrdToFSL")
        volume_out_db = volume_out.replace(image_registration.DATA_FOLDER, "")
        cursor.execute('''UPDATE Images SET filepath = ? WHERE id = ?''', (volume_out_db, img_id))

        if image_type == 'post':
            volume_label = path + '%02d' % case_id + '_post-label.nii'
        elif image_type == 'pre':
            volume_label = path + '%02d' % case_id + '_pre-label.nii'
        if not os.path.exists(volume):
            continue

        cursor.execute('''INSERT INTO Labels(image_id, segmented_by, description) VALUES(?,?,?)''',
                       (img_id, 'Hans Kristian', 'all'))
        label_id = cursor.lastrowid

        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_" + image_type\
            + "_label_all.nii.gz"
        os.system(image_registration.DWICONVERT_PATH + " --inputVolume " + volume_label + " -o " +
                  volume_label_out + " --conversionMode NrrdToFSL")
        volume_label_out_db = volume_label_out.replace(image_registration.DATA_FOLDER, "")
        cursor.execute('''UPDATE Labels SET filepath = ? WHERE id = ?''',
                       (volume_label_out_db, label_id))

        conn.commit()

    cursor.close()
    conn.close()


def save_transform_to_database(data_transforms):
    """ Save data transforms to database"""
    conn = sqlite3.connect(image_registration.DB_PATH)
    conn.text_factory = str
    for moving_image_id, transform in data_transforms:
        print("-----", moving_image_id, transform)

        cursor = conn.execute('''SELECT pid from Images where id = ? ''', (moving_image_id,))
        pid = cursor.fetchone()[0]

        folder = image_registration.DATA_FOLDER + str(pid) + "/registration/"
        shutil.rmtree(folder, True)
        mkdir_p(folder)

        if not isinstance(transform, list):
            transform = [transform]

        transform_paths = ""
        for _transform in transform:
            shutil.move(_transform, folder)
            transform_paths += str(pid) + "/registration/" + basename(_transform) + ", "

        transform_paths = transform_paths[:-2]
        cursor2 = conn.execute('''UPDATE Images SET transform = ? WHERE id = ?''', (transform_paths, moving_image_id))
        cursor2 = conn.execute('''UPDATE Images SET fixed_image = ? WHERE id = ?''', (-1, moving_image_id))

        conn.commit()
        cursor.close()
        cursor2.close()

    conn.close()
    vacuum_db()


def vacuum_db():
    """ Clean up database"""
    conn = sqlite3.connect(image_registration.DB_PATH)
    cursor = conn.execute('''VACUUM; ''')
    cursor.close()
    conn.close()


def get_image_paths(image_ids):
    """ get image paths """
    conn = sqlite3.connect(image_registration.DB_PATH)
    conn.text_factory = str

    res = []
    for _id in image_ids:
        cursor = conn.execute('''SELECT transform from Images where id = ? ''', (_id,))
        transform = cursor.fetchone()[0]
        for _transform in transform:
            res.append(_transform)
    cursor.close()
    conn.close()

    return res

if __name__ == "__main__":
    image_registration.setup_paths()

#    try:
#        shutil.rmtree(image_registration.DATA_FOLDER)
#        os.remove("volume_label.nrrd")
#        os.remove("volume.nrrd")
#    except OSError:
#        pass
#    mkdir_p(image_registration.DATA_FOLDER)
#    create_db(image_registration.DB_PATH)

    convert_lisa_data(image_registration.DATA_PATH_LISA, False)
    convert_lisa_data(image_registration.DATA_PATH_LISA_QOL, True)
    convert_annelise_data(image_registration.DATA_PATH_ANNE_LISE)

    convert_lgg_data(image_registration.DATA_PATH_LGG + "PRE_OP/")
    convert_lgg_data(image_registration.DATA_PATH_LGG + "POST/")
    vacuum_db()
