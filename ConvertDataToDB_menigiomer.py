# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:19:49 2016

@author: dahoiv
"""
# pylint: disable= line-too-long
from __future__ import print_function

import glob
import os
import SimpleITK as sitk
import shutil
import sqlite3

import util

MAIN_FOLDER = "/home/dahoiv/disk/data/meningiomer/"
DWICONVERT_PATH = "/home/dahoiv/disk/kode/Slicer/Slicer-SuperBuild/Slicer-build/lib/Slicer-4.6/cli-modules/DWIConvert"


def create_db(path):
    """Make the database"""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE "Images" (
    `id`    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    `pid`    INTEGER,
    `modality`    TEXT,
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
    `comments`    TEXT,
    PRIMARY KEY(pid))''')

    conn.commit()
    cursor.close()
    conn.close()


# pylint: disable= too-many-arguments, too-many-locals
def convert_and_save_dataset(pid, cursor, volume_labels, volume):
    """convert_and_save_dataset"""
    util.mkdir_p(util.DATA_FOLDER + str(pid))
    img_out_folder = util.DATA_FOLDER + str(pid) + "/volumes_labels/"
    util.mkdir_p(img_out_folder)

    cursor.execute('''SELECT pid from Patient where pid = ?''', (pid, ))
    exist = cursor.fetchone()
    if exist is None:
        cursor.execute('''INSERT INTO Patient(pid) VALUES(?)''', (pid, ))

    cursor.execute('''INSERT INTO Images(pid, modality) VALUES(?,?)''',
                   (pid, 'MR'))
    img_id = cursor.lastrowid

    _, file_extension = os.path.splitext(volume)
    volume_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1.nii.gz"
    print("--->", volume_out)

    sitk.WriteImage(sitk.ReadImage(volume), volume_out)
    volume_out_db = volume_out.replace(util.DATA_FOLDER, "")

    cursor.execute('''UPDATE Images SET filepath = ?, filepath_reg = ? WHERE id = ?''', (volume_out_db, None, img_id))
    for volume_label in volume_labels:
        _, file_extension = os.path.splitext(volume_label)
        cursor.execute('''INSERT INTO Labels(image_id, description) VALUES(?,?)''',
                       (img_id, 'all'))
        label_id = cursor.lastrowid

        volume_label_out = img_out_folder + str(pid) + "_" + str(img_id) + "_MR_T1_label_all.nii.gz"
        sitk.WriteImage(sitk.ReadImage(volume_label), volume_label_out)

        volume_label_out_db = volume_label_out.replace(util.DATA_FOLDER, "")
        cursor.execute('''UPDATE Labels SET filepath = ?, filepath_reg = ? WHERE id = ?''',
                       (volume_label_out_db, None, label_id))


def convert_data(path, update=False, case_ids=range(2000)):
    """Convert gbm data """
    # pylint: disable= too-many-locals, too-many-branches, too-many-statements
    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.cursor()

    log = ""

    for case_id in case_ids:
        data_path = path + str(case_id) + "/"
        if not os.path.exists(data_path):
            continue

        pid = str(case_id)
        if update:
            for _file in glob.glob(util.DATA_FOLDER + str(pid) + "/volumes_labels/*"):
                _file = _file.replace(util.DATA_FOLDER, "")
                cursor.execute("DELETE FROM Images WHERE filepath=?", (_file,))
                cursor.execute("DELETE FROM Labels WHERE filepath=?", (_file,))
            try:
                shutil.rmtree(util.DATA_FOLDER + str(pid))
            except OSError:
                pass

        data_path = path + str(case_id) + "/"
        if not os.path.exists(data_path):
            continue

        print(data_path)

        volume_label = glob.glob(data_path + '/*label.mhd')
        if len(volume_label) > 1:
            log = log + "\n Warning!! More than one file with label found "
            for volume_label_i in volume_label:
                log = log + volume_label_i
            continue
        if not volume_label:
            volume_label = glob.glob(data_path + '/*label.nrrd')
            if not volume_label:
                log = log + "\n Warning!! No label found " + data_path
                continue
        volume_label = volume_label[0]

        volume = glob.glob(data_path + '/*vol.mhd')
        if len(volume) > 1:
            log = log + "\n Warning!! More than one file with volume found "
            for volume_i in volume:
                log = log + volume_i
            continue
        if not volume:
            volume = glob.glob(data_path + '/*vol.nrrd')
            if not volume:
                log = log + "\n Warning!! No volume found " + data_path
                continue
        volume = volume[0]
        if not os.path.exists(volume):
            log = log + "\n Warning!! No volume found " + data_path

        convert_and_save_dataset(pid, cursor, [volume_label], volume)
        conn.commit()

    with open("Log.txt", "w") as text_file:
        text_file.write(log)
    cursor.close()
    conn.close()


def vacuum_db():
    """ Clean up database"""
    conn = sqlite3.connect(util.DB_PATH)
    cursor = conn.execute('''VACUUM; ''')
    cursor.close()
    conn.close()


if __name__ == "__main__":
    util.setup_paths("meningiomer")
    try:
        shutil.rmtree(util.DATA_FOLDER)
    except OSError:
        pass
    util.mkdir_p(util.DATA_FOLDER)
    create_db(util.DB_PATH)
    convert_data(MAIN_FOLDER + "org_data/")

    vacuum_db()
