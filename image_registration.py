
""" install
http://neuro.debian.net/install_pkg.html?p=fsl-complete

sudo apt-get install -y libblas-dev liblapack-dev libfreetype6-dev
sudo apt-get install -y cmake ninja-build git
sudo apt-get install gfortran

git clone git://github.com/stnava/ANTs.git
mkdir antsbin
cd antsbin
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ../ANTs/
ninja

sudo apt-get install python-pip
cd
git clone git@github.com:Danielhiversen/NeuroImageRegistration.git
cd NeuroImageRegistration/
virtualenv venv
source venv/bin/activate
sudo pip install --upgrade setuptools
sudo pip install --upgrade distribute
pip install -r requirements.txt

ant registration parameters inspired by
http://miykael.github.io/nipype-beginner-s-guide/normalize.html
https://www.icts.uiowa.edu/
confluence/display/BRAINSPUBLIC/ANTS+conversion+to+antsRegistration+for+same+data+set

"""
# pylint: disable= redefined-builtin
# import nipype.interfaces.dipy as dipy
from __future__ import print_function
from __future__ import division
import gzip
from multiprocessing import Pool
import os
from os.path import basename
import datetime
import sqlite3
import shutil
from builtins import map
from builtins import str
from nilearn.image import resample_img
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nibabel as nib
import numpy as np

from img_data import img_data
import util

MULTITHREAD = 1  # 1,23,4....., "max"
MULTITHREAD = "max"

RIGID = 'rigid'
AFFINE = 'affine'
SYN = 'syn'

BE_METHOD = 2


def pre_process(img, do_bet=True):
    # pylint: disable= too-many-statements, too-many-locals
    """ Pre process the data"""

    path = img.temp_data_path

    input_file = img.img_filepath
    n4_file = path + util.get_basename(input_file) + '_n4.nii.gz'
    norm_file = path + util.get_basename(n4_file) + '_norm.nii.gz'
    resampled_file = path + util.get_basename(norm_file) + '_resample.nii.gz'
    name = util.get_basename(resampled_file) + "_be"
    img.pre_processed_filepath = path + name + '.nii.gz'

    if os.path.exists(img.pre_processed_filepath) and\
       (os.path.exists(path + name + 'Composite.h5') or BE_METHOD == 1):
        if BE_METHOD == 0:
            img.init_transform = path + name + 'Composite.h5'
        util.generate_image(img.pre_processed_filepath, util.TEMPLATE_VOLUME)
        return img

    n4bias = ants.N4BiasFieldCorrection()
    n4bias.inputs.dimension = 3
    n4bias.inputs.input_image = input_file
    n4bias.inputs.output_image = n4_file
    n4bias.run()

    # normalization [0,100], same as template
    normalize_img = nib.load(n4_file)
    temp_data = normalize_img.get_data()
    temp_img = nib.Nifti1Image(temp_data/np.amax(temp_data)*100,
                               normalize_img.affine, normalize_img.header)
    temp_img.to_filename(norm_file)

    # resample volume to 1 mm slices
    target_affine_3x3 = np.eye(3) * 1
    img_3d_affine = resample_img(norm_file, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    if not do_bet:
        img.pre_processed_filepath = resampled_file
        return img

    if BE_METHOD == 0:
        img.init_transform = path + name + '_InitRegTo' + str(img.fixed_image) + '.h5'

        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = True
        reg.inputs.fixed_image = resampled_file
        reg.inputs.moving_image = util.TEMPLATE_VOLUME
        reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.num_threads = 1
        reg.inputs.initial_moving_transform_com = True

        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.metric = ['MI', 'MI']
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.metric_weight = [1, 1]
        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.number_of_iterations = ([[10000, 10000, 10000, 10000],
                                            [10000, 10000, 10000, 10000]])

        reg.inputs.convergence_threshold = [1.e-6]*2
        reg.inputs.shrink_factors = [[9, 5, 3, 1], [9, 5, 3, 1]]
        reg.inputs.smoothing_sigmas = [[8, 4, 1, 0], [8, 4, 1, 0]]
        reg.inputs.transform_parameters = [(0.25,), (0.25,)]
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.use_estimate_learning_rate_once = [True, True]

        reg.inputs.write_composite_transform = True
        reg.inputs.output_transform_prefix = path + name
        reg.inputs.output_warped_image = path + name + '_beReg.nii.gz'

        transform = path + name + 'InverseComposite.h5'
        print("starting be registration")
        reg.run()
        print("Finished be registration")

        reg_volume = util.transform_volume(resampled_file, transform)
        shutil.copy(transform, img.init_transform)

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg_volume
        mult.inputs.second_input = util.TEMPLATE_MASK
        mult.inputs.output_product_image = img.pre_processed_filepath
        mult.run()

        util.generate_image(img.pre_processed_filepath, reg_volume)
    elif BE_METHOD == 1:
        # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
        bet = fsl.BET(command="fsl5.0-bet")
        bet.inputs.in_file = resampled_file
        # pylint: disable= pointless-string-statement
        """ fractional intensity threshold (0->1); default=0.5;
        smaller values give larger brain outline estimates"""
        bet.inputs.frac = 0.25
        """ vertical gradient in fractional intensity threshold (-1->1);
        default=0; positive values give larger brain outline at bottom,
        smaller at top """
        bet.inputs.vertical_gradient = 0
        """  This attempts to reduce image bias, and residual neck voxels.
        This can be useful when running SIENA or SIENAX, for example.
        Various stages involving FAST segmentation-based bias field removal
        and standard-space masking are combined to produce a result which
        can often give better results than just running bet2."""
        # bet.inputs.reduce_bias = True
        bet.inputs.mask = True

        bet.inputs.out_file = img.pre_processed_filepath

        bet.run()
        util.generate_image(img.pre_processed_filepath, resampled_file)
    elif BE_METHOD == 2:
        name = util.get_basename(resampled_file) + "_bet"

        # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
        bet = fsl.BET(command="fsl5.0-bet")
        bet.inputs.in_file = resampled_file
        # pylint: disable= pointless-string-statement
        """ fractional intensity threshold (0->1); default=0.5;
        smaller values give larger brain outline estimates"""
        bet.inputs.frac = 0.1
        """ vertical gradient in fractional intensity threshold (-1->1);
        default=0; positive values give larger brain outline at bottom,
        smaller at top """
        bet.inputs.vertical_gradient = 0
        """  This attempts to reduce image bias, and residual neck voxels.
        This can be useful when running SIENA or SIENAX, for example.
        Various stages involving FAST segmentation-based bias field removal
        and standard-space masking are combined to produce a result which
        can often give better results than just running bet2."""
        bet.inputs.reduce_bias = True
        bet.inputs.mask = True
        bet.inputs.out_file = path + name + '.nii.gz'
        bet.run()

        name = name + "_be"
        pre_processed_filepath1 = path + name + '.nii.gz'
        name = name + "_be"
        img.pre_processed_filepath1 = path + name + '.nii.gz'
        img.init_transform = path + name + '_InitRegTo' + str(img.fixed_image) + '.h5'

        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = True
        reg.inputs.fixed_image = bet.inputs.out_file
        reg.inputs.moving_image = util.TEMPLATE_MASKED_VOLUME
        reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.num_threads = 1
        reg.inputs.initial_moving_transform_com = True

        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.metric = ['MI', 'MI']
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.metric_weight = [1, 1]
        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.sampling_strategy = ['Regular'] * 2
        reg.inputs.sampling_percentage = [0.5] * 2
        reg.inputs.number_of_iterations = ([[10000, 10000, 5000, 5000],
                                            [10000, 10000, 5000, 5000]])
        reg.inputs.shrink_factors = [[9, 5, 3, 1], [9, 5, 3, 1]]
        reg.inputs.smoothing_sigmas = [[8, 4, 1, 0], [8, 4, 1, 0]]
        reg.inputs.transform_parameters = [(0.75,), (0.75,)]
        reg.inputs.convergence_threshold = [1.e-6]*2
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.use_estimate_learning_rate_once = [True, True]

        reg.inputs.write_composite_transform = True
        reg.inputs.output_transform_prefix = path + name
        reg.inputs.output_warped_image = path + name + '_beReg.nii.gz'

        transform = path + name + 'Composite.h5'
        print("starting be registration")
        start_time = datetime.datetime.now()
        reg.run()
        print("Finished be registration1: ")
        print(datetime.datetime.now() - start_time)

        reg_mask = util.transform_volume(util.TEMPLATE_MASK, transform)
        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = resampled_file
        mult.inputs.second_input = reg_mask
        mult.inputs.output_product_image = pre_processed_filepath1
        mult.run()

        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = transform
        reg.inputs.fixed_image = resampled_file
        reg.inputs.moving_image = util.TEMPLATE_MASKED_VOLUME
        reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.num_threads = 1
        reg.inputs.initial_moving_transform = True

        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.metric = ['MI', 'MI']
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.metric_weight = [1, 1]
        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.sampling_strategy = ['Regular'] * 2
        reg.inputs.sampling_percentage = [0.5] * 2
        reg.inputs.number_of_iterations = ([[10000, 10000, 5000, 5000],
                                            [10000, 10000, 5000, 5000]])
        reg.inputs.shrink_factors = [[7, 5, 3, 1], [9, 5, 3, 1]]
        reg.inputs.smoothing_sigmas = [[8, 4, 1, 0], [8, 4, 1, 0]]
        reg.inputs.transform_parameters = [(0.75,), (0.75,)]
        reg.inputs.convergence_threshold = [1.e-6]*2
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.use_estimate_learning_rate_once = [True, True]

        reg.inputs.write_composite_transform = True
        reg.inputs.output_transform_prefix = path + name
        reg.inputs.output_warped_image = path + name + '_beReg.nii.gz'

        transform = path + name + 'InverseComposite.h5'
        print("starting be registration")
        start_time = datetime.datetime.now()
        reg.run()
        print("Finished be registration2: ")
        print(datetime.datetime.now() - start_time)

        reg_volume = util.transform_volume(resampled_file, transform)
        shutil.copy(transform, img.init_transform)

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg_volume
        mult.inputs.second_input = util.TEMPLATE_MASK
        mult.inputs.output_product_image = pre_processed_filepath1
        mult.run()

        util.generate_image(img.pre_processed_filepath, reg_volume)

    print("---BET", img.pre_processed_filepath)
    return img


def registration(moving_img, fixed, reg_type):
    # pylint: disable= too-many-statements
    """Image2Image registration """
    reg = ants.Registration()

    path = moving_img.temp_data_path
    name = util.get_basename(moving_img.pre_processed_filepath) + '_' + reg_type
    moving_img.processed_filepath = path + name + '_RegTo' + str(moving_img.fixed_image) + '.nii.gz'
    moving_img.transform = path + name + '_RegTo' + str(moving_img.fixed_image) + '.h5'

    init_moving_transform = moving_img.init_transform
    if init_moving_transform is not None and os.path.exists(init_moving_transform):
        print("Found initial transform")
        # reg.inputs.initial_moving_transform = init_moving_transform
        reg.inputs.initial_moving_transform_com = False
        mask = util.transform_volume(moving_img.label_inv_filepath, moving_img.init_transform, True)
    else:
        reg.inputs.initial_moving_transform_com = True
        mask = moving_img.label_inv_filepath
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = moving_img.pre_processed_filepath
    reg.inputs.fixed_image_mask = mask
    reg.inputs.moving_image = fixed
    reg.inputs.num_threads = 1
    if reg_type == RIGID:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.metric = ['MI']
        reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.convergence_window_size = [5]
        reg.inputs.number_of_iterations = ([[10000, 10000, 10000, 10000, 10000]])
        reg.inputs.shrink_factors = [[5, 4, 3, 2, 1]]
        reg.inputs.smoothing_sigmas = [[4, 3, 2, 1, 0]]
        reg.inputs.sigma_units = ['vox']
        reg.inputs.transform_parameters = [(0.25,)]
        reg.inputs.use_histogram_matching = [True]
        reg.inputs.metric_weight = [1.0]
    elif reg_type == AFFINE:
        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.metric = ['MI', 'CC']
        reg.inputs.radius_or_number_of_bins = [32, 5]

        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.number_of_iterations = ([[10000, 10000, 1000, 1000, 1000],
                                            [10000, 10000, 1000, 1000, 1000]])
        reg.inputs.shrink_factors = [[5, 4, 3, 2, 1], [5, 4, 3, 2, 1]]
        reg.inputs.smoothing_sigmas = [[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]]
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.transform_parameters = [(0.25,),
                                           (0.15,)]
        reg.inputs.use_histogram_matching = [False, True]
        reg.inputs.metric_weight = [1.0]*2
    elif reg_type == SYN:
        reg.inputs.transforms = ['Affine', 'SyN']
        reg.inputs.metric = ['MI', ['MI', 'CC']]
        reg.inputs.metric_weight = [1]  + [[0.5, 0.5]]
        reg.inputs.radius_or_number_of_bins = [32, [32, 5]]
        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.sampling_strategy = ['Regular']  + [[None, None]]
        reg.inputs.sampling_percentage = [0.5]  + [[None, None]]
        reg.inputs.number_of_iterations = ([[1000, 1000],
                                            [100, 75, 75, 75]])
        reg.inputs.shrink_factors = [[2, 1], [5, 3, 2, 1]]
        reg.inputs.smoothing_sigmas = [[1, 0], [4, 2, 1, 0]]
        reg.inputs.convergence_threshold = [1.e-6]  + [-0.01]
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.transform_parameters = [(0.25,),
                                           (0.15, 3.0, 0.0)]
        reg.inputs.use_estimate_learning_rate_once = [True] * 2
        reg.inputs.use_histogram_matching = [False, True]
    else:
        raise Exception("Wrong registration format " + reg_type)
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.write_composite_transform = True
    reg.inputs.output_transform_prefix = path + name
    transform = path + name + 'InverseComposite.h5'

    if os.path.exists(moving_img.processed_filepath) and\
       os.path.exists(moving_img.transform):
        # generate_image(reg.inputs.output_warped_image, fixed)
        return moving_img
    print("starting registration")
    start_time = datetime.datetime.now()
    reg.run()
    print("Finished registration: ")
    print(datetime.datetime.now() - start_time)

    util.transform_volume(moving_img.pre_processed_filepath, transform,
                          outputpath=moving_img.processed_filepath)
    shutil.copy(transform, moving_img.transform)
    util.generate_image(moving_img.processed_filepath, fixed)

    return moving_img


def process_dataset(args):
    """ pre process and registrate volume"""
    (moving_image_id, reg_type, save_to_db) = args
    print(moving_image_id)

    start_time = datetime.datetime.now()
    img = img_data(moving_image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
    img = pre_process(img)

    print("\n\n\n\n -- Run time preprocess: ")
    print(datetime.datetime.now() - start_time)

    for k in range(3):
        try:
            img = registration(img, util.TEMPLATE_MASKED_VOLUME, reg_type)
            break
        # pylint: disable= broad-except
        except Exception as exp:
            print('Crashed during' + str(k+1) + ' of 3 \n' + str(exp))
    print("\n\n\n\n -- Run time: ")
    print(datetime.datetime.now() - start_time)
    if save_to_db:
        save_transform_to_database([img])
    return img


def get_transforms(moving_dataset_image_ids, reg_type=None,
                   process_dataset_func=process_dataset, save_to_db=False):
    """Calculate transforms """
    if MULTITHREAD > 1:
        if MULTITHREAD == 'max':
            pool = Pool()
        else:
            pool = Pool(MULTITHREAD)
        # http://stackoverflow.com/a/1408476/636384
        result = pool.map_async(process_dataset_func,
                                zip(moving_dataset_image_ids,
                                    [reg_type]*len(moving_dataset_image_ids),
                                    [save_to_db]*len(moving_dataset_image_ids))).get(999999999)
        pool.close()
        pool.join()
    else:
        result = list(map(process_dataset_func, zip(moving_dataset_image_ids,
                                                    [reg_type]*len(moving_dataset_image_ids),
                                                    save_to_db)))
    return result


def move_vol(moving, transform, label_img=False):
    """ Move data with transform """
    if label_img:
        # resample volume to 1 mm slices
        target_affine_3x3 = np.eye(3) * 1
        img_3d_affine = resample_img(moving, target_affine=target_affine_3x3,
                                     interpolation='nearest')
        resampled_file = util.TEMP_FOLDER_PATH + util.get_basename(moving) + '_resample.nii.gz'
        # pylint: disable= no-member
        img_3d_affine.to_filename(resampled_file)

    else:
        img = img_data(-1, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img.set_img_filepath(moving)
        resampled_file = pre_process(img, False).pre_processed_filepath

    result = util.transform_volume(moving, transform, label_img)
    util.generate_image(result, util.TEMPLATE_VOLUME)
    return result


def save_transform_to_database(data_transforms):
    """ Save data transforms to database"""
    # pylint: disable= too-many-locals, bare-except
    conn = sqlite3.connect(util.DB_PATH)
    conn.text_factory = str

    for img in data_transforms:
        cursor = conn.execute('''SELECT pid from Images where id = ? ''', (img.image_id,))
        pid = cursor.fetchone()[0]

        folder = util.DATA_FOLDER + str(pid) + "/registration_transforms/"
        util.mkdir_p(folder)

        transform_paths = ""
        print(img.get_transforms())
        for _transform in img.get_transforms():
            print(_transform)
            dst_file = folder + util.get_basename(_transform) + '.gz'
            if os.path.exists(dst_file):
                os.remove(dst_file)
            with open(_transform, 'rb') as f_in, gzip.open(dst_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            transform_paths += str(pid) + "/registration_transforms/" +\
                basename(_transform) + '.gz' + ", "
        transform_paths = transform_paths[:-2]

        cursor2 = conn.execute('''UPDATE Images SET transform = ? WHERE id = ?''',
                               (transform_paths, img.image_id))
        cursor2 = conn.execute('''UPDATE Images SET fixed_image = ? WHERE id = ?''',
                               (img.fixed_image, img.image_id))

        folder = util.DATA_FOLDER + str(pid) + "/reg_volumes_labels/"
        util.mkdir_p(folder)
        vol_path = util.compress_vol(img.processed_filepath)
        shutil.copy(vol_path, folder)

        volume_db = str(pid) + "/reg_volumes_labels/" + basename(vol_path)
        cursor2 = conn.execute('''UPDATE Images SET filepath_reg = ? WHERE id = ?''',
                               (volume_db, img.image_id))

        cursor = conn.execute('''SELECT filepath, id from Labels where image_id = ? ''',
                              (img.image_id,))
        for (row, label_id) in cursor:
            temp = util.compress_vol(move_vol(util.DATA_FOLDER + row,
                                              img.get_transforms(), True))
            shutil.copy(temp, folder)
            label_db = str(pid) + "/reg_volumes_labels/" + basename(temp)
            cursor2 = conn.execute('''UPDATE Labels SET filepath_reg = ? WHERE id = ?''',
                                   (label_db, label_id))

        conn.commit()
        cursor.close()
        cursor2.close()

#    cursor = conn.execute('''VACUUM; ''')
    conn.close()
