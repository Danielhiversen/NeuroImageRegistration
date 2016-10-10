# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:54:08 2016

@author: dahoiv
"""

from __future__ import print_function
from __future__ import division
import sys
import errno
import gzip
import os
from os.path import basename
from os.path import splitext
import sqlite3
from nilearn import datasets
import nipype.interfaces.ants as ants
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
# pylint: disable= wrong-import-position
import matplotlib.pyplot as plt  # noqa
import matplotlib.cm as cm  # noqa

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'

TEMP_FOLDER_PATH = ""
DATA_FOLDER = ""
DB_PATH = ""

TEMPLATE_VOLUME = datasets.fetch_icbm152_2009(data_dir="./").get("t1")
TEMPLATE_MASK = datasets.fetch_icbm152_2009(data_dir="./").get("mask")
TEMPLATE_MASKED_VOLUME = ""


def setup(temp_path, datatype=""):
    """setup for current computer """
    # pylint: disable= global-statement
    global TEMP_FOLDER_PATH
    TEMP_FOLDER_PATH = temp_path
    mkdir_p(TEMP_FOLDER_PATH)
    setup_paths(datatype)
    prepare_template(TEMPLATE_VOLUME, TEMPLATE_MASK)


def setup_paths(datatype=""):
    """setup for current computer """
    # pylint: disable= global-statement, line-too-long
    if datatype not in ["LGG", "GBM", ""]:
        print("Unkown datatype " + datatype)
        raise Exception

    global DATA_FOLDER, DB_PATH

    hostname = os.uname()[1]
    if hostname == 'dahoiv-Alienware-15':
        DATA_FOLDER = "/mnt/dokumneter/data/database2/"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'
    elif hostname == 'dahoiv-Precision-M6500':
        DATA_FOLDER = "/home/dahoiv/database/"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        DATA_FOLDER = "/media/ingerid/data/daniel/database2/"
        os.environ["PATH"] += os.pathsep + '/home/daniel/antsbin/bin/'
    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " + sys.argv[0])
        raise Exception

    DATA_FOLDER = DATA_FOLDER + datatype + "/"
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


# pylint: disable= dangerous-default-value
def post_calculations(moving_dataset_image_ids, result=dict()):
    """ Transform images and calculate avg"""
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str

    for _id in moving_dataset_image_ids:
        print(_id)
        cursor = conn.execute('''SELECT filepath_reg from Images where id = ? ''', (_id,))
        db_temp = cursor.fetchone()
        if db_temp[0] is None:
            continue
        vol = DATA_FOLDER + db_temp[0]
        print(vol)
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


def get_image_id_and_qol(qol_param):
    """ Get image id and qol """
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT pid from QualityOfLife''')

    image_id = []
    qol = []
    for pid in cursor:
        pid = pid[0]
        _qol = conn.execute("SELECT " + qol_param + " from QualityOfLife where pid = ?",
                            (pid, )).fetchone()[0]
        if _qol is None:
            continue

        _id = conn.execute('''SELECT id from Images where pid = ?''', (pid, )).fetchone()
        if not _id:
            continue
        _id = _id[0]

        image_id.extend([_id])
        qol.extend([_qol*100])
    cursor.close()
    conn.close()

    return (image_id, qol)


def find_seg_images(moving_image_id):
    """ Find segmentation images"""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath_reg, description from Labels where image_id = ? ''',
                          (moving_image_id,))
    images = []
    for (row, label) in cursor:
        images.append((DATA_FOLDER + row, label))

    cursor.close()
    conn.close()
    return images


def transform_volume(vol, transform, label_img=False, outputpath=None, ref_img=TEMPLATE_VOLUME):
    """Transform volume """
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


def sum_calculation(images, label, val=None, save=False, folder=TEMP_FOLDER_PATH):
    """ Calculate sum volumes """
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
        _sum = _sum + np.array(img.get_data())*val_i
        temp = np.array(img.get_data())
        temp[temp != 0] = 1.0
        _total = _total + temp
    if save:
        result_img = nib.Nifti1Image(_sum, img.affine)
        result_img.to_filename(path_n)
        generate_image(path_n, TEMPLATE_VOLUME)

    return (_sum, _total)


def avg_calculation(images, label, val=None, save=False, folder=TEMP_FOLDER_PATH):
    """ Calculate average volumes """
    path = folder + 'avg_' + label + '.nii'
    path = path.replace('label', 'tumor')

    (_sum, _total) = sum_calculation(images, label, val, save=False)
    average = _sum / _total

    if save:
        img = nib.load(images[0])
        result_img = nib.Nifti1Image(average, img.affine)
        result_img.to_filename(path)
        generate_image(path, TEMPLATE_VOLUME)
    return average


def calculate_t_test(images, label, save=False, folder=TEMP_FOLDER_PATH):
    """ Calculate sum volumes """
    path_n = folder + 't-test_.nii'
    path_n = path_n.replace('label', 'tumor')

    _sum = None
    _total = None
    for (file_name, val_i) in zip(images, val):
        if val_i is None:
            continue
        img = nib.load(file_name)
        if _sum is None:
            _sum = np.zeros(img.get_data().shape)
            _total = np.zeros(img.get_data().shape)
        _sum = _sum + np.array(img.get_data())*val_i
        temp = np.array(img.get_data())
        temp[temp != 0] = 1.0
        _total = _total + temp
    if save:
        result_img = nib.Nifti1Image(_sum, img.affine)
        result_img.to_filename(path_n)
        generate_image(path_n, TEMPLATE_VOLUME)

    return (_sum, _total)


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
