# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:54:08 2016

@author: dahoiv
"""

from __future__ import print_function
from __future__ import division
import datetime
import sys
import errno
import gzip
import logging
import os
from os.path import basename
from os.path import splitext
import sqlite3
import time
import multiprocessing
import psutil
from nilearn import datasets
import nipype.interfaces.ants as ants
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy import stats
from scipy.spatial import distance
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
# pylint: disable= wrong-import-position, too-many-lines
import matplotlib.pyplot as plt  # noqa
import matplotlib.cm as cm  # noqa

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

LOGGER = logging.getLogger("NeuroReg")

TEMP_FOLDER_PATH = ""
DATA_FOLDER = ""
DB_PATH = ""
ATLAS_FOLDER_PATH = ""

TEMPLATE_VOLUME = datasets.fetch_icbm152_2009(data_dir="./").get("t1")
TEMPLATE_MASK = datasets.fetch_icbm152_2009(data_dir="./").get("mask")
TEMPLATE_MASKED_VOLUME = ""


def setup(temp_path, data="glioma"):
    """setup for current computer """
    # pylint: disable= global-statement
    LOGGER.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler('log.txt', maxBytes=5*10000000,
                                                        backupCount=5)
    file_handler.set_name('log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.set_name('consoleHamdler')
    # console_handler.setLevel(logging.INFO)
    # ch_format = logging.Formatter('%(asctime)s - %(message)s')
    # console_handler.setFormatter(ch_format)
    # LOGGER.addHandler(console_handler)

    global TEMP_FOLDER_PATH
    TEMP_FOLDER_PATH = temp_path
    mkdir_p(TEMP_FOLDER_PATH)
    setup_paths(data)
    prepare_template(TEMPLATE_VOLUME, TEMPLATE_MASK)


def setup_paths(data="glioma"):
    """setup for current computer """
    # pylint: disable= global-statement, line-too-long, too-many-branches, too-many-statements
    global DATA_FOLDER, DB_PATH, ATLAS_FOLDER_PATH, BRAINSResample_PATH

    hostname = os.uname()[1]
    if hostname == 'dddd':
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'
        BRAINSResample_PATH = '/home/dahoiv/disk/kode/Slicer/Slicer-SuperBuild/Slicer-build/lib/Slicer-4.6/cli-modules/BRAINSResample'
    elif hostname == 'dahoiv-Precision-M6500':
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        os.environ["PATH"] += os.pathsep + '/home/leb/dev/ANTs/antsbin/bin'
        ATLAS_FOLDER_PATH = '/media/leb/data/Atlas/'
        BRAINSResample_PATH = '/home/leb/dev/BRAINSTools/build/bin/BRAINSResample'
    elif hostname == 'medtech-beast':
        os.environ["PATH"] += os.pathsep + '/home/leb/dev/ANTs/antsbin/bin'
        ATLAS_FOLDER_PATH = '/home/leb/data/Atlas/'
        BRAINSResample_PATH = '/home/leb/dev/BRAINSTools/build/bin/BRAINSResample'
    elif hostname == 'SINTEF-0ZQHTDG':
        os.environ["PATH"] += os.pathsep + '/Users/leb/dev/ANTs/antsbin/bin'
        ATLAS_FOLDER_PATH = '/Users/leb/OneDrive - SINTEF/Prosjekter/Nevro/Brain atlas/Atlases'
    elif 'unity' in hostname or 'compute' in hostname:
        os.environ["PATH"] += os.pathsep + '/home/danieli/antsbin/bin/' + os.pathsep + '/home/danieli/antsbin/bin/'
    else:
        LOGGER.error("Unkown host name " + hostname)
        LOGGER.error("Add your host name path to " + sys.argv[0])
        raise Exception

    if data == 'glioma':
        if hostname == 'dddd':
            DATA_FOLDER = "/home/dahoiv/disk/data/Segmentations/database/"
        elif hostname == 'dahoiv-Precision-M6500':
            DATA_FOLDER = "/home/dahoiv/database/"
        elif hostname == 'ingerid-PC':
            DATA_FOLDER = "/media/leb/data/database/"
        elif hostname == 'medtech-beast':
            DATA_FOLDER = "/home/leb/data/database/"
        elif hostname == 'SINTEF-0ZQHTDG':
            DATA_FOLDER = "/Volumes/Neuro/Segmentations/database/"
        elif 'unity' in hostname or 'compute' in hostname:
            DATA_FOLDER = '/work/danieli/neuro_data/database/'
        else:
            LOGGER.error("Unkown host name " + hostname)
            LOGGER.error("Add your host name path to " + sys.argv[0])
            raise Exception
    elif data == 'MolekylareMarkorer':
        if hostname == 'dddd':
            DATA_FOLDER = "/home/dahoiv/disk/data/MolekylareMarkorer/database_MM/"
        elif hostname == 'dahoiv-Precision-M6500':
            DATA_FOLDER = ""
        elif hostname == 'ingerid-PC':
            DATA_FOLDER = "/media/leb/data/database_MM/"
        elif hostname == 'medtech-beast':
            DATA_FOLDER = "/home/leb/data/database_MM/"
        elif hostname == 'SINTEF-0ZQHTDG':
            DATA_FOLDER = "/Volumes/Neuro/Segmentations/database_MM"
        elif 'unity' in hostname or 'compute' in hostname:
            DATA_FOLDER = '/work/danieli/database_MM/'
        else:
            LOGGER.error("Unkown host name " + hostname)
            LOGGER.error("Add your host name path to " + sys.argv[0])
            raise Exception
    elif data == 'meningiomer':
        if hostname == 'dddd':
            DATA_FOLDER = "/home/dahoiv/disk/data/meningiomer/database_meningiomer/"
        elif 'unity' in hostname or 'compute' in hostname:
            DATA_FOLDER = '/work/danieli/database_meningiomer/'
        else:
            LOGGER.error("Unkown host name " + hostname)
            LOGGER.error("Add your host name path to " + sys.argv[0])
            raise Exception
    else:
        LOGGER.error("Unkown data type " + data)
        raise Exception

    DB_PATH = DATA_FOLDER + "brainSegmentation.db"


def prepare_template(template_vol, template_mask, overwrite=False):
    """ prepare template volumemoving"""
    # pylint: disable= global-statement,
    global TEMPLATE_MASKED_VOLUME

    TEMPLATE_MASKED_VOLUME = TEMP_FOLDER_PATH + "masked_template.nii"
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = template_vol
    mult.inputs.second_input = template_mask
    mult.inputs.output_product_image = TEMPLATE_MASKED_VOLUME
    if not overwrite and os.path.exists(mult.inputs.output_product_image):
        return
    mult.run()


def post_calculations(moving_dataset_image_ids, result=None):
    """ Transform images and calculate avg"""
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    if result is None:
        result = {}

    for _id in moving_dataset_image_ids:
        cursor = conn.execute('''SELECT filepath_reg from Images where id = ? ''', (_id,))
        db_temp = cursor.fetchone()
        if db_temp[0] is None:
            LOGGER.error("No volume data for image_id " + str(_id))
            continue
        vol = DATA_FOLDER + db_temp[0]
        label = "img"
        if label in result:
            result[label].append(vol)
        else:
            result[label] = [vol]

        for (segmentation, label) in find_reg_label_images(_id):
            if label in result:
                result[label].append(segmentation)
            else:
                result[label] = [segmentation]

        cursor.close()
    conn.close()
    return result


def get_transforms_from_db(img_id, conn):
    """Get transforms from the database """
    cursor = conn.execute('''SELECT transform, fixed_image from Images where id = ? ''', (img_id,))
    db_temp = cursor.fetchone()

    fixed_image_id = db_temp[1]
    if fixed_image_id > 0:
        transforms = get_transforms_from_db(fixed_image_id, conn)
    else:
        transforms = []
    if db_temp[0] is None:
        return []
    img_transforms = db_temp[0].split(",")
    for _transform in img_transforms:
        transforms.append(DATA_FOLDER + _transform.strip())
    return transforms


def get_image_id_and_qol(qol_param, exclude_pid=None, glioma_grades=None):
    """ Get image id and qol """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from QualityOfLife''')
    if not glioma_grades:
        glioma_grades = [2, 3, 4]

    image_id = []
    qol = []
    for pid in cursor:
        pid = pid[0]
        if exclude_pid and pid in exclude_pid:
            continue
        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            LOGGER.error("---No data for " + str(pid) + " " + str(qol_param))
            continue
        _id = _id[0]
        _glioma_grade = conn.execute('''SELECT glioma_grade from Patient where pid = ?''',
                                     (pid, )).fetchone()
        if not _glioma_grade:
            LOGGER.error("No glioma_grade for " + str(pid) + " " + str(qol_param))
            continue
        if _glioma_grade[0] not in glioma_grades:
            continue
        if qol_param:
            _qol = conn.execute("SELECT " + qol_param + " from QualityOfLife where pid = ?",
                                (pid, )).fetchone()[0]
            if _qol is None:
                LOGGER.error("No qol data for " + str(_id) + " " + str(qol_param))
                continue
            qol.extend([_qol])
        image_id.extend([_id])
    cursor.close()
    conn.close()

    return image_id, qol


def get_qol(image_ids, qol_param):
    """ Get image id and qol """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    qol = []
    image_ids_with_qol = []
    pids = []
    for image_id in image_ids:
        _pid = conn.execute('''SELECT pid from Images where id = ?''', (image_id, )).fetchone()
        if not _pid:
            LOGGER.error("---No data for " + str(_pid) + " " + str(qol_param))
            continue
        pid = _pid[0]
        if qol_param:
            _qol = conn.execute("SELECT " + qol_param + " from QualityOfLife where pid = ?",
                                (pid, )).fetchone()
            if _qol is None or _qol[0] is None:
                LOGGER.error("No qol data for " + str(pid) + " " + str(qol_param))
                continue
            qol.extend([_qol[0]])
            pids.append(pid)
            image_ids_with_qol.extend([image_id])

    conn.close()
    return image_ids_with_qol, qol


def get_tumor_volume(image_ids):
    """ Get image id and tumor volume """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    volumes = []
    image_ids_with_volume = []
    for image_id in image_ids:
        _volume = conn.execute("SELECT tumor_volume from Images where id = ?",
                               (image_id, )).fetchone()
        if _volume is None or _volume[0] is None:
            LOGGER.error("No qol data for " + str(image_id))
            continue
        volumes.extend([_volume[0]])
        image_ids_with_volume.extend([image_id])

    conn.close()
    return image_ids_with_volume, volumes


def get_image_id_and_survival_days(
        study_id=None,
        exclude_pid=None,
        glioma_grades=None,
        registration_date_upper_lim=None,
        censor_date_str=None,
        survival_group=None,
        resection=False):
    """ Get image id and qol
    :param study_id: string with ID of study to be included
    :param exclude_pid: list of patient IDs to be excluded
    :param glioma_grades: list of glioma grades to be included
    :param registration_date_upper_lim: string with date limiting how new registrations should be included
    :param censor_date_str: string with censor date used for patients that are still alive
    :param survival_group: list of survival groups, each group given by a list with two elements representing the lower and upper limits in days
    :return: image_id: image IDs for all included patients
    :return: survival_days: survival days up until censor date for all included patients
    """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')

    image_id = []
    survival_days = []
    for pid in cursor:
        pid = pid[0]

        if exclude_pid and pid in exclude_pid:
            continue

        if study_id:
            _study_id = conn.execute('''SELECT study_id from Patient where pid = ?''',
                                     (pid, )).fetchone()
            if not _study_id[0]:
                continue
            elif study_id not in _study_id[0]:
                continue

        if glioma_grades:
            _glioma_grade = conn.execute('''SELECT glioma_grade from Patient where pid = ?''',
                                     (pid, )).fetchone()
            if not _glioma_grade:
                LOGGER.error("No glioma_grade for PID = " + str(pid))
                continue
            if _glioma_grade[0] not in glioma_grades:
                continue

        if resection:
            _resection = conn.execute('''SELECT resection from Patient where pid = ?''',
                                     (pid, )).fetchone()
            if not _resection:
                LOGGER.error("No resection status for PID = " + str(pid))
                continue
            if _resection[0] == 0:
                continue

        _survival_days = conn.execute("SELECT survival_days from Patient where pid = ?",
                                      (pid, )).fetchone()[0]
        if _survival_days is None:
            _operation_date_str = conn.execute("SELECT op_date from Patient where pid = ?",
                                                 (pid, )).fetchone()[0]

            if _operation_date_str and censor_date_str:
                _operation_date = datetime.datetime.strptime(_operation_date_str[0:10],'%Y-%m-%d')
                _censor_date = datetime.datetime.strptime(censor_date_str,'%Y-%m-%d')
                _survival_days = (_censor_date-_operation_date).days
                if _survival_days < 0:
                    LOGGER.error("Operation date is after censor date for PID = " + str(pid))
                else:
                    print('PID ' + str(pid) + ' is still alive. Survival at censor date: ' + str(_survival_days))

            else:
                LOGGER.error("No survival_days or op_date/censor_date data for PID = " + str(pid))
                continue

        if survival_group and not survival_group[0] <= _survival_days <= survival_group[1]:
            continue

        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            LOGGER.error("No image data for PID = " + str(pid))
            continue
        _id = _id[0]

        if registration_date_upper_lim:
            _reg_date_str = conn.execute('''SELECT registration_date from Images where id = ?''',
                                     (_id, )).fetchone()
            if _reg_date_str:
                _reg_date = datetime.datetime.strptime(_reg_date_str,'%Y-%m-%d')
                _date_upper_lim = datetime.datetime.strptime(registration_date_upper_lim,'%Y-%m-%d')
                if _reg_date > _date_upper_lim:
                    LOGGER.info("Image with ID = " + str(_id) + "has a recent registration date and is excluded")
                    continue

        survival_days.extend([_survival_days])
        image_id.extend([_id])
    cursor.close()
    conn.close()

    return image_id, survival_days


def get_pids_and_image_ids(study_id=None, exclude_pid=None, glioma_grades=None):
    """ Get image PIDs and image IDs for a given study ID and/or glioma grade """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')

    pids = []
    image_ids = []
    for pid in cursor:
        pid = pid[0]

        if exclude_pid and pid in exclude_pid:
            continue

        if study_id:
            _study_id = conn.execute('''SELECT study_id from Patient where pid = ?''',
                                     (pid, )).fetchone()
            if not _study_id[0]:
                continue
            elif study_id not in _study_id[0]:
                continue

        if glioma_grades:
            _glioma_grade = conn.execute('''SELECT glioma_grade from Patient where pid = ?''',
                                     (pid, )).fetchone()
            if not _glioma_grade:
                LOGGER.error("No glioma_grade for " + str(pid))
                continue
            if _glioma_grade[0] not in glioma_grades:
                continue

        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            LOGGER.error("---No data for " + str(pid))
            continue
        _id = _id[0]

        pids.extend([pid])
        image_ids.extend([_id])
    cursor.close()
    conn.close()

    return pids, image_ids


def find_seg_images(moving_image_id):
    """ Find segmentation images"""
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath, description from Labels where image_id = ? ''',
                          (moving_image_id,))
    images = []
    for (row, label) in cursor:
        images.append((DATA_FOLDER + row, label))

    cursor.close()
    conn.close()
    return images


def find_reg_label_images(moving_image_id):
    """ Find reg segmentation images"""
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath_reg, description from Labels where image_id = ? ''',
                          (moving_image_id,))
    images = []
    for (row, label) in cursor:
        images.append((DATA_FOLDER + row, label))

    cursor.close()
    conn.close()
    return images


def transform_volume(vol, transform, label_img=False, outputpath=None, ref_img=None):
    """Transform volume """
    if not ref_img:
        ref_img = TEMPLATE_VOLUME
    transforms = []
    for _transform in ensure_list(transform):
        transforms.append(decompress_file(_transform))

    if outputpath:
        result = outputpath
    else:
        result = TEMP_FOLDER_PATH + get_basename(basename(vol)) + '_reg.nii.gz'
    apply_transforms = ants.ApplyTransforms()
    if label_img:
        apply_transforms.inputs.interpolation = 'NearestNeighbor'
    else:
        apply_transforms.inputs.interpolation = 'Linear'
    apply_transforms.inputs.dimension = 3
    apply_transforms.inputs.input_image = vol
    apply_transforms.inputs.reference_image = ref_img
    apply_transforms.inputs.output_image = result
    apply_transforms.inputs.default_value = 0
    apply_transforms.inputs.transforms = transforms
    apply_transforms.inputs.invert_transform_flags = [False]*len(transforms)
    apply_transforms.run()

    return apply_transforms.inputs.output_image


# pylint: disable= too-many-arguments
def sum_calculation(images, label, val=None, save=False, folder=None, default_value=None):
    """ Calculate sum volumes """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path_n = folder + 'total_' + label + '.nii'
    path_n = path_n.replace('label', 'tumor')

    if not val:
        val = [1]*len(images)

    _sum = None
    _total = None
    for (file_name, val_i) in zip(images, val):
        if val_i is None:
            continue
        img = nib.load(file_name)
        if _sum is None:
            _sum = np.zeros(img.get_data().shape)
            _total = np.zeros(img.get_data().shape)
        temp = np.array(img.get_data())
        _sum += temp*val_i
        temp[temp != 0] = 1.0
        _total += temp
    if default_value is not None:
        _sum[_sum == 0] = default_value

    if save:
        result_img = nib.Nifti1Image(_sum, img.affine)
        result_img.to_filename(path_n)
        generate_image(path_n, TEMPLATE_VOLUME)

    return _sum, _total


# pylint: disable= too-many-arguments
def avg_calculation(images, label, val=None, save=False, folder=None,
                    save_sum=False, default_value=0):
    """ Calculate average volumes """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 'avg_' + label + '.nii'

    (_sum, _total) = sum_calculation(images, label, val, save=save_sum)
    _total[_total == 0] = np.inf
    if val:
        average = _sum / _total
    else:
        average = _sum / len(images)
    average[_total == np.inf] = default_value

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(average, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)
    return average


def mortality_rate_calculation(images, label, survival_days, save=False, folder=None,
                    save_sum=False, default_value=0, max_value=None, per_year=False):
    """ Calculate average volumes """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 'mortality_rate' + label + '.nii'

    (_sum, _total) = sum_calculation(images, 'survival_days', survival_days, save=save_sum)
    _sum[_sum == 0] = np.inf
    mortality_rate = _total / _sum
    mortality_rate[_sum == np.inf] = default_value
    if per_year:
        mortality_rate[mortality_rate>default_value] *= 36525 # Ganger med hundre for å få større verdier

    mortality_rate_pos = mortality_rate[mortality_rate>default_value]

    percentiles = np.zeros(3)
    percentiles[0] = np.percentile(mortality_rate_pos,25)
    percentiles[1] = np.percentile(mortality_rate_pos,50)
    percentiles[2] = np.percentile(mortality_rate_pos,75)

    if max_value:
        mortality_rate[mortality_rate>max_value] = max_value

    img = nib.load(TEMPLATE_MASK)
    mask = img.get_data()
    mortality_rate[mask==0] = default_value

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(mortality_rate, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)

        percentiles_file = open(path[:-4]+'_percentiles.txt','w')
        percentiles_file.write(np.array2string(percentiles, precision=2, separator=', '))
        percentiles_file.close()

        # nifti_reader = vtk.vtkNIFTIImageReader()
        # nifti_reader.SetFileName(path)
        # nifti_reader.Update()
        # vtk_image = nifti_reader.GetOutput()
        # mhd_writer = vtk.vtkMetaImageWriter()
        # mhd_writer.SetInputData(vtk_image)
        # mhd_writer.SetFileName(path[:-3]+'mhd')
        # mhd_writer.Write()

    return mortality_rate


def median_calculation(images, label, val=None, save=False, folder=None, default_value=0):
    """ Calculate median volumes """
    # pylint: disable= too-many-locals, invalid-name
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 'median_' + label + '.nii'

    if not val:
        val = [1]*len(images)

    total = {}
    shape = None
    for (file_name, val_i) in zip(images, val):
        img = nib.load(file_name)
        if shape is None:
            shape = img.get_data().shape
        label_idx = np.where(img.get_data() == 1)
        for (k, l, m) in zip(label_idx[0], label_idx[1], label_idx[2]):
            key = str(k) + "_" + str(l) + "_" + str(m)
            if key in total:
                total[key].append(val_i)
            else:
                total[key] = [val_i]

    median = np.zeros(shape) + default_value

    for key, val_i in total.items():
        _temp = key.split("_")
        k = int(_temp[0])
        m = int(_temp[1])
        n = int(_temp[2])
        median[k, m, n] = np.median(val_i)

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(median, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(median, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)


def std_calculation(images, label, val=None, save=False, folder=None):
    """ Calculate std volume """
    if not folder:
        folder = TEMP_FOLDER_PATH

    if not val:
        val = [1]*len(images)

    (_sum, _total) = sum_calculation(images, label, val, save=False)
    avg_img = _sum / _total
    path = folder + 'std_' + label + '.nii'

    _std = None
    _total = None
    for (file_name, val_i) in zip(images, val):
        if val_i is None:
            continue
        img = nib.load(file_name)
        if _std is None:
            _std = np.zeros(img.get_data().shape)
            _total = np.zeros(img.get_data().shape)
        temp = np.array(img.get_data())
        _std += (temp*val_i - avg_img)**2
        temp[temp != 0] = 1.0
        _total += temp
    _std /= _total

    if save:
        result_img = nib.Nifti1Image(_std, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)

    return _std


# pylint: disable= too-many-arguments
def calc_resection_prob(images_pre, images_post, label, save=False, folder=None, default_value=0):
    """ Calculate average volumes """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 'resection_prob_' + label + '.nii'
    path = path.replace('label', 'tumor')
    path = path.replace('all', 'tumor')
    path = path.replace('img', 'volume')

    val = None
    (_, _total_pre) = sum_calculation(images_pre, label, val, save=False)
    (_, _total_post) = sum_calculation(images_post, label, val, save=False)
    indx = _total_pre == 0
    _total_pre[indx] = np.inf
    res = 1 - _total_post / _total_pre
    res[indx] = default_value

    if save:
        img = nib.load(images_pre[0])
        result_img = nib.Nifti1Image(res, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)
    return res


def calculate_t_test(images, mu_h0, label='Index_value', save=True, folder=None):
    """ Calculate t-test volume """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 't-test.nii'

    (_sum, _total) = sum_calculation(images, label, save=False)
    _std = std_calculation(images, label, save=True)

    temp = mu_h0 - _sum / _total
    # temp[temp<0] = 0
    t_img = (temp) / _std * np.sqrt(_total)

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(t_img, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)

    return t_img


def vlsm(label_paths, label, stat_func, val=None, folder=None,
         n_permutations=0, alternative='less'):
    """ Calculate average volumes """
    # pylint: disable= too-many-locals, invalid-name, too-many-branches

    if not folder:
        folder = TEMP_FOLDER_PATH

    total = {}
    _id = 0
    for file_name in label_paths:
        img = nib.load(file_name)
        label_idx = np.where(img.get_data() == 1)
        for (k, l, m) in zip(label_idx[0], label_idx[1], label_idx[2]):
            key = str(k) + "_" + str(l) + "_" + str(m)
            if key in total:
                total[key].append(_id)
            else:
                total[key] = [_id]
        _id += 1
    shape = img.get_data().shape

    res = permutation_test(total, val, shape, alternative, stat_func)
    path = folder + 'p-val_' + label + '.nii'
    path = path.replace('label', 'tumor')
    img = nib.load(label_paths[0])
    result_img = nib.Nifti1Image(res['p_val'], img.affine)
    result_img.to_filename(path)
    generate_image(path, TEMPLATE_VOLUME)
    if n_permutations == 0:
        return

    queue = multiprocessing.Queue()
    jobs = []
    total_res = np.ones((shape[0], shape[1], shape[2]))

    def _help_permutation_test(index, total, values, shape, alternative, stat_func):
        permutation_res = permutation_test(total, values, shape,
                                           alternative, stat_func)['statistic']
        queue.put((index, permutation_res))

    processes = multiprocessing.cpu_count()
    nr_of_jobs = 0
    finished_jobs = 0
    index = 0
    values = list(val)
    while finished_jobs < n_permutations:
        if nr_of_jobs < processes and index < n_permutations\
                and psutil.virtual_memory().percent < 90:
            nr_of_jobs += 1
            # pylint: disable= no-member
            np.random.shuffle(values)
            process = multiprocessing.Process(target=_help_permutation_test,
                                              args=[index, total, values, shape, None, stat_func])
            process.start()
            jobs.append(process)
            index += 1

        if not queue.empty():
            (_index, permutation_res) = queue.get()
            jobs[_index].join()

            temp = np.zeros((shape[0], shape[1], shape[2]))
            if alternative == 'less':
                idx = permutation_res > res['statistic']
            else:
                idx = permutation_res < res['statistic']
            temp[idx] = 1.0 / (n_permutations + 1)
            total_res[(idx & (total_res == 1))] = 0
            total_res += temp

            nr_of_jobs -= 1
            finished_jobs += 1
            LOGGER.info(str(finished_jobs / n_permutations))
        if not nr_of_jobs < processes and queue.empty():
            time.sleep(2)

    path = folder + 'p-val_permutations_' + label + '.nii'
    path = path.replace('label', 'tumor')
    img = nib.load(label_paths[0])
    result_img = nib.Nifti1Image(total_res, img.affine)
    result_img.to_filename(path)
    generate_image(path, TEMPLATE_VOLUME)


def permutation_test(total, values, shape, alternative, stat_func):
    """Do permutation test."""
    # pylint: disable= too-many-locals, invalid-name
    start_time = datetime.datetime.now()
    res = {}
    if alternative is not None:
        res['p_val'] = np.zeros(shape) + 1
    res['statistic'] = np.zeros(shape)

    for key, vox_ids in total.items():
        if len(vox_ids) < 2:
            continue
        _temp = key.split("_")
        k = int(_temp[0])
        m = int(_temp[1])
        n = int(_temp[2])
        group1 = [values[index] for index in vox_ids]
        ids = range(len(values))
        group2 = [values[index] for index in ids if index not in vox_ids]
        (p_val, statistic) = stat_func(group1, group2, alternative)
        if alternative is not None:
            res['p_val'][k, m, n] = p_val
        res['statistic'][k, m, n] = statistic

    LOGGER.info(str(datetime.datetime.now() - start_time))

    return res


def brunner_munzel_test(x, y, alternative='less'):
    """
    Computes the Brunner Munzel statistic

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    x : sequence
        Input
    y : sequence
        Input
    alternative : {greater, less, two_sided }

    Returns
    -------
    statistic : float
        The Brunner Munzel  statistics
    pvalue : float
        Approximate p-value assuming a t distribution.

    http://codegists.com/snippet/python/brunner_munzel_testpy_katsuyaito_python

     """
    # pylint: disable= too-many-locals, invalid-name
    x = np.ma.asarray(x).compressed().view(np.ndarray)
    y = np.ma.asarray(y).compressed().view(np.ndarray)
    ranks = stats.rankdata(np.concatenate([x, y]))
    (nx, ny) = (len(x), len(y))
    rankx = stats.rankdata(x)
    ranky = stats.rankdata(y)
    rank_mean1 = np.mean(ranks[0:nx])
    rank_mean2 = np.mean(ranks[nx:nx+ny])

    v1_set = [(i - j - rank_mean1 + (nx + 1)/2)**2 for (i, j) in zip(ranks[0:nx], rankx)]
    v2_set = [(i - j - rank_mean2 + (ny + 1)/2)**2 for (i, j) in zip(ranks[nx:nx+ny], ranky)]

    v1 = np.sum(v1_set)/(nx - 1)
    v2 = np.sum(v2_set)/(ny - 1)
    statistic = nx * ny * (rank_mean2 - rank_mean1)/(nx + ny)/np.sqrt(nx * v1 + ny * v2)
    if alternative is None:
        return (-1, statistic)

    dfbm = ((nx * v1 + ny * v2)**2)/(((nx * v1)**2)/(nx - 1) + ((ny * v2)**2)/(ny - 1))
    if (alternative == "greater") | (alternative == "g"):
        prob = stats.t.cdf(statistic, dfbm)
    elif (alternative == "less") | (alternative == "l"):
        prob = 1 - stats.t.cdf(statistic, dfbm)
    else:
        abst = np.abs(statistic)
        prob = stats.t.cdf(abst, dfbm)
        prob = 2 * min(prob, 1 - prob)

    return (prob, statistic)


def mannwhitneyu_test(x, y, alternative='less'):
    """
    Computes the Mann-Whitney statistic

    Parameters
    ----------
    x : sequence
        Input
    y : sequence
        Input
    alternative : {greater, less, two_sided }

    Returns
    -------
    statistic : float
        The Mann-Whitney statistics
    pvalue : float
        Approximate p-value assuming a t distribution.

     """
    # pylint: disable= invalid-name
    if alternative is None:
        statistic, prob = stats.mannwhitneyu(x, y, alternative='less')
        prob = -1
    else:
        statistic, prob = stats.mannwhitneyu(x, y, alternative=alternative)
    return (prob, -1 * statistic)


def generate_image(path, path2, out_path=None):
    """ generate png images"""
    def show_slices(slices, layers):
        """ Show 2d slices"""
        _, axes = plt.subplots(1, len(slices))
        for i, slice_i in enumerate(slices):
            # pylint: disable= no-member
            if layers:
                axes[i].imshow(layers[i].T, cmap="gray", origin="lower")
            axes[i].imshow(slice_i.T, cmap=cm.Reds, origin="lower", alpha=0.6)

    # pylint: disable= invalid-name
    img = nib.load(path).get_data()
    if len(img.shape) > 3:
        img = img[:, :, :, 0]
    x = int(img.shape[0]/2)
    y = int(img.shape[1]/2)
    z = int(img.shape[2]/2)
    slice_0 = img[x, :, :]
    slice_1 = img[:, y, :]
    slice_2 = img[:, :, z]
    slices = [slice_0, slice_1, slice_2]

    if path2:
        img_template = nib.load(path2).get_data()
        x = int(img_template.shape[0]/2)
        y = int(img_template.shape[1]/2)
        z = int(img_template.shape[2]/2)
        slice_0 = img_template[x, :, :]
        slice_1 = img_template[:, y, :]
        slice_2 = img_template[:, :, z]
        slices_template = [slice_0, slice_1, slice_2]
    else:
        slices_template = None

    show_slices(slices, slices_template)
    name = get_basename(out_path) if out_path else get_basename(path)
    plt.suptitle(name)

    out_path = out_path if out_path else splitext(splitext(path)[0])[0] + ".png"
    plt.savefig(out_path)
    plt.close()


def compress_vol(vol):
    """Compress volume"""
    if vol[-3:] == ".gz":
        return vol
    res = vol + ".gz"
    temp = nib.load(vol)
    temp.to_filename(res)
    return res


def decompress_file(gzip_path):
    """Decompress file """
    gzip_path = gzip_path.strip()
    if gzip_path[-3:] != '.gz':
        return gzip_path

    in_file = gzip.open(gzip_path, 'rb')
    # uncompress the gzip_path INTO THE 'data' variable
    data = in_file.read()
    in_file.close()

    # get gzip filename (without directories)
    gzip_fname = os.path.basename(gzip_path)
    # get original filename (remove 3 characters from the end: ".gz")
    fname = gzip_fname[:-3]
    uncompressed_path = os.path.join(TEMP_FOLDER_PATH, fname)

    # store uncompressed file data from 'data' variable
    open(uncompressed_path, 'w').write(data)

    return uncompressed_path


def ensure_list(value):
    """Wrap value in list if it is not one."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def mkdir_p(path):
    """Make new folder if not exits"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_basename(filepath):
    """Get basename of filepath"""
    return splitext(splitext(basename(filepath))[0])[0]


def get_center_of_mass(filepath,label=None):
    """Get center_of_mass of filepath"""
    img = nib.load(filepath)
    if label:
        data = img.get_data() == label
    else:
        data = img.get_data()
    com = ndimage.measurements.center_of_mass(data)
    com_idx = [int(_com) for _com in com]

    spacing = img.header.get_zooms()
    qform = img.header.get_qform(coded=True)
    if not qform[1]:
        sform = img.header.get_sform(coded=True)
        if not sform[1]:
            LOGGER.error('The file ' + filepath + ' contains no QForm or SForm matrix.')
            raise Exception
        else:
            qform = sform
            LOGGER.warning('The file ' + filepath + ' contains no QForm matrix. Using SForm matrix instead.')
    trans = [qform[0][0, 3], qform[0][1, 3], qform[0][2, 3]]

    res = [c*s for (c, s) in zip(com, spacing)]
    com = [r+t for (r, t) in zip(res, trans)]
    return com, com_idx

def get_label_coordinates(filepath,label=None):
    img = nib.load(filepath)
    if label:
        data = img.get_data() == label
    else:
        data = img.get_data()
    dims = data.shape
    spacing = img.header.get_zooms()
    qform = img.header.get_qform(coded=True)
    if not qform[1]:
        sform = img.header.get_sform(coded=True)
        if not sform[1]:
            LOGGER.error('The file ' + filepath + ' contains no QForm or SForm matrix.')
            raise Exception
        else:
            qform = sform
            LOGGER.warning('The file ' + filepath + ' contains no QForm matrix. Using SForm matrix instead.')
    trans = [qform[0][0, 3], qform[0][1, 3], qform[0][2, 3]]

    label_coordinates = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if data[i,j,k]:
                    idx = [i, j, k]
                    res = [c*s for (c, s) in zip(idx, spacing)]
                    coordinates = [r+t for (r, t) in zip(res, trans)]
                    label_coordinates.append(coordinates)
    return label_coordinates


def get_surface(filepath,labelRange=[1,1]):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filepath)
    reader.Update()

    labelmap = reader.GetOutput()

    qform_transform = vtk.vtkTransform()
    qform_matrix = reader.GetQFormMatrix()
    if not qform_matrix:
        sform_matrix = reader.GetSFormMatrix()
        if not sform_matrix:
            LOGGER.error('The file ' + filepath + ' contains no QForm or SForm matrix.')
            raise Exception
        else:
            qform_matrix = sform_matrix
            LOGGER.warning('The file ' + filepath + ' contains no QForm matrix. Using SForm matrix instead.')
    qform_transform.SetMatrix(qform_matrix)

    # Find surface of label map
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(reader.GetOutputPort())
    dmc.GenerateValues(1, labelRange)
    dmc.Update()

    # Transform surface to patient coordinates
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetTransform(qform_transform)
    tpd.SetInputConnection(dmc.GetOutputPort())
    tpd.Update()

    # Convert surface points to Python list
    point_cloud = vtk_to_numpy(tpd.GetOutput().GetPoints().GetData()).tolist()
    surface = {
        'point_cloud': point_cloud,
        'labelmap': labelmap,
        'qform_transform': qform_transform
    }

    return surface


def get_min_distance(surface,points):
    """
    Find minimum distance between a single point/point cloud and a surface.

    Parameters
    ----------
    surface : dict
        Dictionary containing a labelmap, a point_cloud and a transform
        as produced by get_surface()
    points : nested list
        Coordinates of points as a nested list.

    Returns
    -------
    min_dist : double
        Minimum distance in millimeter. If the number is negative, the points
        are inside the surface.
        NB! If there are multiple points and the distances are very small,
            it is likely that there are points both inside and outside surface.

    """
    def point_is_inside(surface,point):
        """
        Check if a point is inside or outside of a surface.

        Parameters
        ----------
        surface : dict
            Dictionary containing a labelmap, a point_cloud and a transform
            as produced by get_surface()
        point : [x,y,z]
            Coordinates of point that will be checked.

        Returns
        -------
        point_is_inside : {1,-1}
             1 if point is outside of surface.
            -1 if point is inside of surface.

        """

        if len(point) is not 3:
            LOGGER.error('Point does not have exactly three coordinates')
            raise Exception

        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)

        # Transform points to image coordinates
        tpd_inverse = vtk.vtkTransformPolyDataFilter()
        tpd_inverse.SetTransform(surface['qform_transform'].GetInverse())
        tpd_inverse.SetInputData(pointsPolydata)
        tpd_inverse.Update()

        # Get scalar values
        pf = vtk.vtkProbeFilter()
        pf.SetSourceData(surface['labelmap'])
        pf.SetInputConnection(tpd_inverse.GetOutputPort())
        pf.Update()
        label_at_point = vtk_to_numpy(pf.GetOutput().GetPointData().GetScalars()).tolist()
        point_is_inside = 1 - 2*(label_at_point[0] > 0)

        return point_is_inside

    point_is_inside = point_is_inside(surface,points[0])
    min_dist = distance.cdist(surface['point_cloud'], points, 'euclidean').min()#*point_is_inside
    return min_dist


def write_fcsv(name_out, folder_out, tag_data, color, glyph_type):
    """Write fcsv file, https://www.slicer.org/wiki/Modules:Fiducials-Documentation-3.6"""
    fscv_data = '# Markups fiducial file version = 4.7' + os.linesep
    fscv_data += '# CoordinateSystem = 0' + os.linesep
    fscv_data += '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID'
    fscv_data += os.linesep

    for val in tag_data:
        fscv_data += val['Name'] + "," + val['PositionGlobal'] + ",0,0,0,1,1,1,1,"
        fscv_data += val['Name'] + "," + val.get("desc", "") + "," + os.linesep
    fcsv_file = open(folder_out + name_out + ".fcsv", 'w')
    fcsv_file.write(fscv_data)
    fcsv_file.close()

    mrml_text = '<MRML  version="Slicer4.4.0" userTags="">' + os.linesep + \
                ' <MarkupsDisplay id="vtkMRMLMarkupsDisplayNode1" name="MarkupsDisplay" color="'\
                + color + '" selectedColor="' + color\
                + '" textScale="2" glyphScale="3" glyphType="'\
                + str(glyph_type) + '" sliceProjection="7" sliceProjectionColor="1 1 1" '\
                + 'sliceProjectionOpacity="0.6" />' + os.linesep\
                + ' <MarkupsFiducial id="vtkMRMLMarkupsFiducialNode1" name="' + name_out\
                + '" references="display:vtkMRMLMarkupsDisplayNode1;storage:'\
                + 'vtkMRMLMarkupsFiducialStorageNode1;" />' + os.linesep\
                + ' <MarkupsFiducialStorage id="vtkMRMLMarkupsFiducialStorageNode1" '\
                + 'name="MarkupsFiducialStorage" fileName="'\
                + name_out + u'.fcsv" coordinateSystem="0" />' + os.linesep\
                + '</MRML>' + os.linesep

    mrml_file = open(folder_out + name_out + ".mrml", 'w')
    mrml_file.write(mrml_text)
    mrml_file.close()


def get_label_defs():
    """Get brain area defs"""
    label_defs = dict()
    label_defs[30] = "frontal left"
    label_defs[210] = "frontal left"

    label_defs[17] = "frontal right"
    label_defs[211] = "frontal right"

    label_defs[57] = "parietal left"
    label_defs[6] = "parietal left"

    label_defs[105] = "parietal right"
    label_defs[2] = "parietal right"

    label_defs[83] = "temporal left"
    label_defs[218] = "temporal left"
    label_defs[59] = "temporal right"
    label_defs[219] = "temporal right"

    label_defs[73] = "occipital left"
    label_defs[8] = "occipital left"
    label_defs[45] = "occipital right"
    label_defs[4] = "occipital right"

    label_defs[67] = "cerebellum left"
    label_defs[76] = "cerebellum right"
    label_defs[20] = "brainstem"

    label_defs[3] = "deep central"
    label_defs[9] = "deep central"
    label_defs[232] = "deep central"
    label_defs[233] = "deep central"
    label_defs[255] = "deep central"

    label_defs[39] = "deep central"
    label_defs[53] = "deep central"

    label_defs[14] = "deep central"
    label_defs[16] = "deep central"

    label_defs[102] = "deep central"
    label_defs[203] = "deep central"

    label_defs[33] = "deep central"
    label_defs[23] = "deep central"

    label_defs[12] = "deep central"
    label_defs[11] = "deep central"

    label_defs[29] = "deep central"
    label_defs[254] = "deep central"

    label_defs[0] = "deep central"

    label_defs[28] = "skull"

    return label_defs


def get_right_left_label_defs():
    """Get brain area defs"""
    label_defs = dict()
    label_defs[30] = "left"
    label_defs[210] = "left"
    label_defs[17] = "right"
    label_defs[211] = "right"

    label_defs[83] = "left"
    label_defs[218] = "left"
    label_defs[59] = "right"
    label_defs[219] = "right"

    label_defs[57] = "left"
    label_defs[6] = "left"
    label_defs[105] = "right"
    label_defs[2] = "right"

    label_defs[73] = "left"
    label_defs[8] = "left"
    label_defs[45] = "right"
    label_defs[4] = "right"

    label_defs[67] = "left"
    label_defs[76] = "right"
    label_defs[20] = "unknown"

    label_defs[3] = "left"
    label_defs[9] = "right"
    label_defs[232] = "unknown"
    label_defs[233] = "unknown"
    label_defs[255] = "unknown"

    label_defs[39] = "left"
    label_defs[53] = "right"

    label_defs[14] = "left"
    label_defs[16] = "right"

    label_defs[102] = "left"
    label_defs[203] = "right"

    label_defs[33] = "left"
    label_defs[23] = "right"

    label_defs[12] = "left"
    label_defs[11] = "right"

    label_defs[29] = "left"
    label_defs[254] = "right"

    label_defs[0] = "unknown"

    label_defs[28] = "skull"

    return label_defs


def get_label_defs_hammers_mith():
    """Get brain area defs"""
    # pylint: disable= too-many-statements
    label_defs = dict()
    label_defs[1] = "TL hippocampus R"
    label_defs[2] = "TL hippocampus L"
    label_defs[3] = "TL amygdala R"
    label_defs[4] = "TL amygdala L"
    label_defs[5] = "TL anterior temporal lobe medial part R"
    label_defs[6] = "TL anterior temporal lobe medial part L"
    label_defs[7] = "TL anterior temporal lobe lateral part R"
    label_defs[8] = "TL anterior temporal lobe lateral part L"
    label_defs[9] = "TL parahippocampal and ambient gyrus R"
    label_defs[10] = "TL parahippocampal and ambient gyrus L"
    label_defs[11] = "TL superior temporal gyrus middle part R"
    label_defs[12] = "TL superior temporal gyrus middle part L"
    label_defs[13] = "TL middle and inferior temporal gyrus R"
    label_defs[14] = "TL middle and inferior temporal gyrus L"
    label_defs[15] = "TL fusiform gyrus R"
    label_defs[16] = "TL fusiform gyrus L"
    label_defs[17] = "cerebellum R"
    label_defs[18] = "cerebellum L"
    label_defs[19] = "brainstem excluding substantia nigra"
    label_defs[20] = "insula posterior long gyrus L"
    label_defs[21] = "insula posterior long gyrus R"
    label_defs[22] = "OL lateral remainder occipital lobe L"
    label_defs[23] = "OL lateral remainder occipital lobe R"
    label_defs[24] = "CG anterior cingulate gyrus L"
    label_defs[25] = "CG anterior cingulate gyrus R"
    label_defs[26] = "CG posterior cingulate gyrus L"
    label_defs[27] = "CG posterior cingulate gyrus R"
    label_defs[28] = "FL middle frontal gyrus L"
    label_defs[29] = "FL middle frontal gyrus R"
    label_defs[30] = "TL posterior temporal lobe L"
    label_defs[31] = "TL posterior temporal lobe R"
    label_defs[32] = "PL angular gyrus L"
    label_defs[33] = "PL angular gyrus R"
    label_defs[34] = "caudate nucleus L"
    label_defs[35] = "caudate nucleus R"
    label_defs[36] = "nucleus accumbens L"
    label_defs[37] = "nucleus accumbens R"
    label_defs[38] = "putamen L"
    label_defs[39] = "putamen R"
    label_defs[40] = "thalamus L"
    label_defs[41] = "thalamus R"
    label_defs[42] = "pallidum L"
    label_defs[43] = "pallidum R"
    label_defs[44] = "corpus callosum"
    label_defs[45] = "Lateral ventricle excluding temporal horn R"
    label_defs[46] = "Lateral ventricle excluding temporal horn L"
    label_defs[47] = "Lateral ventricle temporal horn R"
    label_defs[48] = "Lateral ventricle temporal horn L"
    label_defs[49] = "Third ventricle"
    label_defs[50] = "FL precentral gyrus L"
    label_defs[51] = "FL precentral gyrus R"
    label_defs[52] = "FL straight gyrus L"
    label_defs[53] = "FL straight gyrus R"
    label_defs[54] = "FL anterior orbital gyrus L"
    label_defs[55] = "FL anterior orbital gyrus R"
    label_defs[56] = "FL inferior frontal gyrus L"
    label_defs[57] = "FL inferior frontal gyrus R"
    label_defs[58] = "FL superior frontal gyrus L"
    label_defs[59] = "FL superior frontal gyrus R"
    label_defs[60] = "PL postcentral gyrus L"
    label_defs[61] = "PL postcentral gyrus R"
    label_defs[62] = "PL superior parietal gyrus L"
    label_defs[63] = "PL superior parietal gyrus R"
    label_defs[64] = "OL lingual gyrus L"
    label_defs[65] = "OL lingual gyrus R"
    label_defs[66] = "OL cuneus L"
    label_defs[67] = "OL cuneus R"
    label_defs[68] = "FL medial orbital gyrus L"
    label_defs[69] = "FL medial orbital gyrus R"
    label_defs[70] = "FL lateral orbital gyrus L"
    label_defs[71] = "FL lateral orbital gyrus R"
    label_defs[72] = "FL posterior orbital gyrus L"
    label_defs[73] = "FL posterior orbital gyrus R"
    label_defs[74] = "substantia nigra L"
    label_defs[75] = "substantia nigra R"
    label_defs[76] = "FL subgenual frontal cortex L"
    label_defs[77] = "FL subgenual frontal cortex R"
    label_defs[78] = "FL subcallosal area L"
    label_defs[79] = "FL subcallosal area R"
    label_defs[80] = "FL pre-subgenual frontal cortex L"
    label_defs[81] = "FL pre-subgenual frontal cortex R"
    label_defs[82] = "TL superior temporal gyrus anterior part L"
    label_defs[83] = "TL superior temporal gyrus anterior part R"
    label_defs[84] = "PL supramarginal gyrus L"
    label_defs[85] = "PL supramarginal gyrus R"
    label_defs[86] = "insula anterior short gyrus L"
    label_defs[87] = "insula anterior short gyrus R"
    label_defs[88] = "insula middle short gyrus L"
    label_defs[89] = "insula middle short gyrus R"
    label_defs[90] = "insula posterior short gyrus L"
    label_defs[91] = "insula posterior short gyrus R"
    label_defs[92] = "insula anterior inferior cortex L"
    label_defs[93] = "insula anterior inferior cortex R"
    label_defs[94] = "insula anterior long gyrus L"
    label_defs[95] = "insula anterior long gyrus R"

    return label_defs
