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
from nilearn import datasets
import nipype.interfaces.ants as ants
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy import stats
import matplotlib
matplotlib.use('Agg')
# pylint: disable= wrong-import-position
import matplotlib.pyplot as plt  # noqa
import matplotlib.cm as cm  # noqa

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

LOGGER = logging.getLogger("NeuroReg")

TEMP_FOLDER_PATH = ""
DATA_FOLDER = ""
DB_PATH = ""

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
    # pylint: disable= global-statement, line-too-long, too-many-branches
    global DATA_FOLDER, DB_PATH

    hostname = os.uname()[1]
    if hostname == 'dddd':
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'
    elif hostname == 'dahoiv-Precision-M6500':
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        os.environ["PATH"] += os.pathsep + '/home/daniel/antsbin/bin/'
    elif 'unity' in hostname or 'compute' in hostname:
        os.environ["PATH"] += os.pathsep + '/home/danieli/antsbin/bin/' + os.pathsep + '/home/danieli/antsbin/bin/'
    else:
        LOGGER.error("Unkown host name " + hostname)
        LOGGER.error("Add your host name path to " + sys.argv[0])
        raise Exception

    if data == 'glioma':
        if hostname == 'dddd':
            DATA_FOLDER = "/home/dahoiv/disk/data/Segmentations/unity/database/"
        elif hostname == 'dahoiv-Precision-M6500':
            DATA_FOLDER = "/home/dahoiv/database/"
        elif hostname == 'ingerid-PC':
            DATA_FOLDER = "/media/ingerid/data/daniel/database4/"
        elif 'unity' in hostname or 'compute' in hostname:
            DATA_FOLDER = '/work/danieli/neuro_data/database/'
        else:
            LOGGER.error("Unkown data " + data)
            raise Exception
    elif data == 'MolekylareMarkorer':
        if hostname == 'dahoiv-Alienware-15':
            DATA_FOLDER = "/home/dahoiv/disk/data/MolekylareMarkorer/database_MM3/"
        elif hostname == 'dahoiv-Precision-M6500':
            DATA_FOLDER = ""
        elif hostname == 'ingerid-PC':
            DATA_FOLDER = "/media/ingerid/data/daniel/database_MM/"
        else:
            LOGGER.error("Unkown data " + data)
            raise Exception
    else:
        LOGGER.error("Unkown data type " + data)
        raise Exception

    DB_PATH = DATA_FOLDER + "brainSegmentation.db"


def prepare_template(template_vol, template_mask):
    """ prepare template volumemoving"""
    # pylint: disable= global-statement,
    global TEMPLATE_MASKED_VOLUME

    TEMPLATE_MASKED_VOLUME = TEMP_FOLDER_PATH + "masked_template.nii"
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = template_vol
    mult.inputs.second_input = template_mask
    mult.inputs.output_product_image = TEMPLATE_MASKED_VOLUME
    if os.path.exists(mult.inputs.output_product_image):
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
            LOGGER.error("No volume data for image_id " + _id)
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

    return (image_id, qol)


def get_qol(image_ids, qol_param):
    """ Get image id and qol """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    qol = []
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
    conn.close()

    return qol

def get_image_id_and_survival_days(exclude_pid=None, glioma_grades=None):
    """ Get image id and qol """
    conn = sqlite3.connect(DB_PATH, timeout=120)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from Patient''')
    if not glioma_grades:
        glioma_grades = [2, 3, 4]

    image_id = []
    survival_days = []
    for pid in cursor:
        pid = pid[0]
        if exclude_pid and pid in exclude_pid:
            continue
        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            LOGGER.error("---No data for " + str(pid))
            continue
        _id = _id[0]
        _glioma_grade = conn.execute('''SELECT glioma_grade from Patient where pid = ?''',
                                     (pid, )).fetchone()
        if not _glioma_grade:
            LOGGER.error("No glioma_grade for " + str(pid))
            continue
        if _glioma_grade[0] not in glioma_grades:
            continue

        _survival_days = conn.execute("SELECT survival_days from Patient where pid = ?",
                                      (pid, )).fetchone()[0]
        if _survival_days is None:
            LOGGER.error("No survival_days data for " + str(_id))
            continue
        survival_days.extend([_survival_days])
        image_id.extend([_id])
    cursor.close()
    conn.close()

    return (image_id, survival_days)


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
def sum_calculation(images, label, val=None, save=False, folder=None, default_value=0):
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
        _sum = _sum + temp*val_i
        temp[temp != 0] = 1.0
        _total = _total + temp
    _sum[_sum == 0] = default_value

    if save:
        result_img = nib.Nifti1Image(_sum, img.affine)
        result_img.to_filename(path_n)
        generate_image(path_n, TEMPLATE_VOLUME)

    return (_sum, _total)


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
        _std = _std + (temp*val_i - avg_img)**2
        temp[temp != 0] = 1.0
        _total = _total + temp
    _std = _std / _total

    if save:
        result_img = nib.Nifti1Image(_std, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)

    return _std


# pylint: disable= too-many-arguments
def avg_calculation(images, label, val=None, save=False, folder=None,
                    save_sum=False, default_value=0):
    """ Calculate average volumes """
    if not folder:
        folder = TEMP_FOLDER_PATH
    path = folder + 'avg_' + label + '.nii'
    path = path.replace('label', 'tumor')

    (_sum, _total) = sum_calculation(images, label, val, save=save_sum, default_value=default_value)
    _total[_total == 0] = np.inf
    if val:
        average = _sum / _total
    else:
        average = _sum / len(images)

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(average, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)
    return average


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


def vlsm(label_paths, label, stat_func, val=None, folder=None, n_permutations=0):
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
        _id = _id + 1
    shape = img.get_data().shape

    res = permutation_test(total, val, shape, 'less', stat_func)
    path = folder + 'vlsm_' + label + '.nii'
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

    def _help_permutation_test(index, total, values, shape, alternative):
        permutation_res = permutation_test(total, values, shape,
                                           alternative, stat_func)['statistic']
        queue.put((index, permutation_res))

    processes = 1 #multiprocessing.cpu_count()
    nr_of_jobs = 0
    finished_jobs = 0
    index = 0
    values = list(val)
    while finished_jobs < n_permutations:
        if nr_of_jobs < processes and index < n_permutations:
            nr_of_jobs = nr_of_jobs + 1
            # pylint: disable= no-member
            np.random.shuffle(values)
            process = multiprocessing.Process(target=_help_permutation_test,
                                              args=[index, total, values, shape, None, stat_func])
            process.start()
            jobs.append(process)
            index = index + 1

        if not queue.empty():
            (_index, permutation_res) = queue.get()
            jobs[_index].join()

            temp = np.zeros((shape[0], shape[1], shape[2]))
            idx = permutation_res > res['statistic']
            temp[idx] = 1.0 / (n_permutations + 1)
            total_res[(idx & (total_res == 1))] = 0
            total_res += temp

            nr_of_jobs = nr_of_jobs - 1
            finished_jobs = finished_jobs + 1
            LOGGER.info(str(finished_jobs / n_permutations))
        if not nr_of_jobs < processes and queue.empty():
            time.sleep(2)

    path = folder + 'vlsm_permutations_' + label + '.nii'
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

    for key, vox_ids in total.iteritems():
        if len(vox_ids) < 2:
            continue
        _temp = key.split("_")
        k = int(_temp[0])
        l = int(_temp[1])
        m = int(_temp[2])
        for index in vox_ids:
            print(index)
            print(values[index])
        group1 = [values[index] for index in vox_ids]
        ids = range(len(values))
        group2 = [values[index] for index in ids if index not in vox_ids]
        (p_val, statistic) = stat_func(group1, group2, alternative)
        if alternative is not None:
            res['p_val'][k, l, m] = p_val
        res['statistic'][k, l, m] = statistic

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
        alternative = "two_sided"
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
    statistic, prob = stats.mannwhitneyu(x, y, alternative=alternative)
    return (prob, statistic)


def generate_image(path, path2, out_path=None):
    """ generate png images"""
    img = nib.load(path).get_data()
    img_template = nib.load(path2).get_data()

    def show_slices(slices, layers):
        """ Show 2d slices"""
        _, axes = plt.subplots(1, len(slices))
        for i, slice_i in enumerate(slices):
            # pylint: disable= no-member
            axes[i].imshow(layers[i].T, cmap="gray", origin="lower")
            axes[i].imshow(slice_i.T, cmap=cm.Reds, origin="lower", alpha=0.6)

    # pylint: disable= invalid-name
    x = int(img.shape[0]/2)
    y = int(img.shape[1]/2)
    z = int(img.shape[2]/2)
    slice_0 = img[x, :, :]
    slice_1 = img[:, y, :]
    slice_2 = img[:, :, z]
    slices = [slice_0, slice_1, slice_2]

    x = int(img_template.shape[0]/2)
    y = int(img_template.shape[1]/2)
    z = int(img_template.shape[2]/2)
    slice_0 = img_template[x, :, :]
    slice_1 = img_template[:, y, :]
    slice_2 = img_template[:, :, z]
    slices_template = [slice_0, slice_1, slice_2]

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
    if gzip_path[:-3] != '.gz':
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


def get_center_of_mass(filepath):
    """Get center_of_mass of filepath"""
    img = nib.load(filepath)
    com = ndimage.measurements.center_of_mass(img.get_data())
    qform = img.header.get_qform()
    spacing = img.header.get_zooms()
    res = [c*s for (c, s) in zip(com, spacing)]
    trans = [qform[0, 3], qform[1, 3], qform[2, 3]]
    res = [r+t for (r, t) in zip(res, trans)]
    return res


def write_fcsv(filepath_out, tag_data, color):
    """Write fcsv file, https://www.slicer.org/wiki/Modules:Fiducials-Documentation-3.6"""
    fscv_data = '# Markups fiducial file version = 4.4' + os.linesep
    fscv_data += '# visibility = 1' + os.linesep
    fscv_data += '# color = ' + color + os.linesep
    fscv_data += '# selectedColor = ' + color + os.linesep
    fscv_data += '# locked = 1'
    fscv_data += '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID'
    fscv_data += os.linesep

    for val in tag_data:
        fscv_data += val['Name'] + "," + val['PositionGlobal'] + ",0,0,0,1,1,1,0,"
        fscv_data += val['Name'] + "," + val.get("desc", "") + "," + os.linesep
    fcsv_file = open(filepath_out, 'w')
    fcsv_file.write(fscv_data)
    fcsv_file.close()
