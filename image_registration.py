
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
from builtins import str
from nilearn.image import resample_img
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nibabel as nib
import numpy as np

from img_data import img_data
import util

RIGID = 'rigid'
AFFINE = 'affine'
SYN = 'syn'
COMPOSITEAFFINE = 'compositeaffine'

BET_FRAC = 0.1
BE_METHOD = 2

HOSTNAME = os.uname()[1]
if 'unity' in HOSTNAME or 'compute' in HOSTNAME:
    NUM_THREADS_ANTS = 6
    MULTITHREAD = 8
    BET_COMMAND = "/home/danieli/fsl/bin/bet"
else:
    NUM_THREADS_ANTS = 4
    # MULTITHREAD = 1  # 1,23,4....., "max"
    MULTITHREAD = "max"
    BET_COMMAND = "fsl5.0-bet"


def pre_process(img, do_bet=True, slice_size=1, reg_type=None, be_method=None):
    # pylint: disable= too-many-statements, too-many-locals, too-many-branches
    """ Pre process the data"""
    path = img.temp_data_path

    input_file = img.img_filepath
    n4_file = path + util.get_basename(input_file) + '_n4.nii.gz'
    norm_file = path + util.get_basename(n4_file) + '_norm.nii.gz'
    resampled_file = path + util.get_basename(norm_file) + '_resample.nii.gz'
    name = util.get_basename(resampled_file) + "_be"
    img.pre_processed_filepath = path + name + '.nii.gz'

    n4bias = ants.N4BiasFieldCorrection()
    n4bias.inputs.dimension = 3
    n4bias.inputs.num_threads = NUM_THREADS_ANTS
    n4bias.inputs.input_image = input_file
    n4bias.inputs.output_image = n4_file
    n4bias.run()

    # normalization [0,100], same as template
    normalize_img = nib.load(n4_file)
    temp_data = normalize_img.get_data()
    temp_img = nib.Nifti1Image(temp_data/np.amax(temp_data)*100,
                               normalize_img.affine, normalize_img.header)
    temp_img.to_filename(norm_file)
    del temp_img

    # resample volume to 1 mm slices
    target_affine_3x3 = np.eye(3) * slice_size
    img_3d_affine = resample_img(norm_file, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    if not do_bet:
        img.pre_processed_filepath = resampled_file
        return img

    if be_method == 0:
        img.init_transform = path + name + '_InitRegTo' + str(img.fixed_image) + '.h5'

        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = True
        reg.inputs.fixed_image = resampled_file
        reg.inputs.moving_image = util.TEMPLATE_VOLUME
        reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.num_threads = NUM_THREADS_ANTS
        reg.inputs.initial_moving_transform_com = True

        if reg_type == RIGID:
            reg.inputs.transforms = ['Rigid', 'Rigid']
        elif reg_type == COMPOSITEAFFINE:
            reg.inputs.transforms = ['Rigid', 'CompositeAffine']
        else:
            reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.metric = ['MI', 'MI']
        reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.metric_weight = [1, 1]
        reg.inputs.convergence_window_size = [5, 5]
        reg.inputs.number_of_iterations = ([[15000, 12000, 10000, 10000, 10000, 5000, 5000],
                                            [10000, 10000, 5000, 5000]])
        reg.inputs.shrink_factors = [[19, 16, 12, 9, 5, 3, 1], [9, 5, 3, 1]]
        reg.inputs.smoothing_sigmas = [[10, 10, 10, 8, 4, 1, 0], [8, 4, 1, 0]]
        reg.inputs.convergence_threshold = [1.e-6]*2
        reg.inputs.transform_parameters = [(0.25,), (0.25,)]
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.use_estimate_learning_rate_once = [True, True]

        reg.inputs.write_composite_transform = True
        reg.inputs.output_transform_prefix = path + name
        reg.inputs.output_warped_image = path + name + '_beReg.nii.gz'

        transform = path + name + 'InverseComposite.h5'
        util.LOGGER.info("starting be registration")
        reg.run()
        util.LOGGER.info("Finished be registration")

        reg_volume = util.transform_volume(resampled_file, transform)
        shutil.copy(transform, img.init_transform)

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg_volume
        mult.inputs.second_input = util.TEMPLATE_MASK
        mult.inputs.output_product_image = img.pre_processed_filepath
        mult.run()

        util.generate_image(img.pre_processed_filepath, reg_volume)
    elif be_method == 1:
        # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
        bet = fsl.BET(command=BET_COMMAND)
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
    elif be_method == 2:
        if BET_FRAC > 0:
            name = util.get_basename(resampled_file) + "_bet"
            # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
            bet = fsl.BET(command=BET_COMMAND)
            bet.inputs.in_file = resampled_file
            # pylint: disable= pointless-string-statement
            """ fractional intensity threshold (0->1); default=0.5;
            smaller values give larger brain outline estimates"""
            bet.inputs.frac = BET_FRAC
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
            util.LOGGER.info("starting bet registration")
            start_time = datetime.datetime.now()
            util.LOGGER.info(bet.cmdline)
            if not os.path.exists(bet.inputs.out_file):
                bet.run()
            util.LOGGER.info("Finished bet registration 0: ")
            util.LOGGER.info(datetime.datetime.now() - start_time)
            name += "_be"
            moving_image = util.TEMPLATE_MASKED_VOLUME
            fixed_image = bet.inputs.out_file
        else:
            name = util.get_basename(resampled_file) + "_be"
            moving_image = util.TEMPLATE_VOLUME
            fixed_image = resampled_file

        img.init_transform = path + name + '_InitRegTo' + str(img.fixed_image) + '.h5'
        img.pre_processed_filepath = path + name + '.nii.gz'
        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = True
        reg.inputs.fixed_image = fixed_image
        reg.inputs.moving_image = moving_image
        reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.num_threads = NUM_THREADS_ANTS
        reg.inputs.initial_moving_transform_com = True

        if reg_type == RIGID:
            reg.inputs.transforms = ['Rigid', 'Rigid']
        elif reg_type == COMPOSITEAFFINE:
            reg.inputs.transforms = ['Rigid', 'CompositeAffine']
        elif reg_type == AFFINE:
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
        reg.inputs.transform_parameters = [(0.25,), (0.25,)]
        reg.inputs.convergence_threshold = [1.e-6]*2
        reg.inputs.sigma_units = ['vox']*2
        reg.inputs.use_estimate_learning_rate_once = [True, True]

        reg.inputs.write_composite_transform = True
        reg.inputs.output_transform_prefix = path + name
        reg.inputs.output_warped_image = path + name + '_TemplateReg.nii.gz'

        transform = path + name + 'InverseComposite.h5'
        util.LOGGER.info("starting be registration")
        util.LOGGER.info(reg.cmdline)
        start_time = datetime.datetime.now()
        if not os.path.exists(reg.inputs.output_warped_image):
            reg.run()
        util.LOGGER.info("Finished be registration: ")
        util.LOGGER.info(datetime.datetime.now() - start_time)

        reg_volume = util.transform_volume(resampled_file, transform)
        shutil.copy(transform, img.init_transform)

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg_volume
        mult.inputs.second_input = util.TEMPLATE_MASK
        mult.inputs.output_product_image = img.pre_processed_filepath
        mult.run()

        util.generate_image(img.pre_processed_filepath, reg_volume)
    else:
        util.LOGGER.error(" INVALID BE METHOD!!!!")

    util.LOGGER.info("---BET " + img.pre_processed_filepath)
    return img


def registration(moving_img, fixed, reg_type):
    # pylint: disable= too-many-statements, too-many-branches
    """Image2Image registration """
    reg = ants.Registration()

    path = moving_img.temp_data_path
    name = util.get_basename(moving_img.pre_processed_filepath) + '_' + reg_type
    moving_img.processed_filepath = path + name + '_RegTo' + str(moving_img.fixed_image) + '.nii.gz'
    moving_img.transform = path + name + '_RegTo' + str(moving_img.fixed_image) + '.h5'

    init_moving_transform = moving_img.init_transform
    if init_moving_transform is not None and os.path.exists(init_moving_transform):
        util.LOGGER.info("Found initial transform")
        # reg.inputs.initial_moving_transform = init_moving_transform
        reg.inputs.initial_moving_transform_com = False
        mask = util.transform_volume(moving_img.label_inv_filepath,
                                     moving_img.init_transform, label_img=True)
    else:
        reg.inputs.initial_moving_transform_com = True
        mask = moving_img.label_inv_filepath
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = moving_img.pre_processed_filepath
    reg.inputs.fixed_image_mask = mask
    reg.inputs.moving_image = fixed
    reg.inputs.num_threads = NUM_THREADS_ANTS
    if reg_type == RIGID:
        reg.inputs.transforms = ['Rigid', 'Rigid', 'Rigid']
        reg.inputs.metric = ['MI', 'MI', 'MI']
        reg.inputs.metric_weight = [1] * 2 + [1]
        reg.inputs.radius_or_number_of_bins = [32, 32, 32]
        reg.inputs.convergence_window_size = [5, 5, 5]
        reg.inputs.sampling_strategy = ['Regular'] * 2 + [None]
        reg.inputs.sampling_percentage = [0.5] * 2 + [None]
        if reg.inputs.initial_moving_transform_com:
            reg.inputs.number_of_iterations = ([[10000, 10000, 10000, 1000, 1000, 1000],
                                                [10000, 10000, 1000, 1000, 1000],
                                                [75, 50, 50]])
            reg.inputs.shrink_factors = [[12, 9, 5, 3, 2, 1], [5, 4, 3, 2, 1], [3, 2, 1]]
            reg.inputs.smoothing_sigmas = [[9, 8, 4, 2, 1, 0], [4, 3, 2, 1, 0], [2, 1, 0]]
        else:
            reg.inputs.number_of_iterations = ([[5000, 5000, 1000, 500],
                                                [5000, 5000, 1000, 500],
                                                [75, 50]])
            reg.inputs.shrink_factors = [[7, 5, 2, 1], [4, 3, 2, 1], [2, 1]]
            reg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [3, 2, 1, 0], [0.5, 0]]
        reg.inputs.convergence_threshold = [1.e-6] * 3
        reg.inputs.sigma_units = ['vox']*3
        reg.inputs.transform_parameters = [(0.25,),
                                           (0.25,),
                                           (0.25,)]
        reg.inputs.use_estimate_learning_rate_once = [True] * 3
        reg.inputs.use_histogram_matching = [False, False, True]
    elif reg_type == AFFINE or reg_type == COMPOSITEAFFINE:
        if reg_type == AFFINE:
            reg.inputs.transforms = ['Rigid', 'Affine', 'Affine']
        else:
            reg.inputs.transforms = ['Rigid', 'CompositeAffine', 'CompositeAffine']
        reg.inputs.metric = ['MI', 'MI', 'MI']
        reg.inputs.metric_weight = [1] * 2 + [1]
        reg.inputs.radius_or_number_of_bins = [32, 32, 32]
        reg.inputs.convergence_window_size = [5, 5, 5]
        reg.inputs.sampling_strategy = ['Regular'] * 2 + [None]
        reg.inputs.sampling_percentage = [0.5] * 2 + [None]
        if reg.inputs.initial_moving_transform_com:
            reg.inputs.number_of_iterations = ([[10000, 10000, 1000, 1000, 1000],
                                                [10000, 10000, 1000, 1000, 1000],
                                                [75, 50, 50]])
            reg.inputs.shrink_factors = [[9, 5, 3, 2, 1], [5, 4, 3, 2, 1], [3, 2, 1]]
            reg.inputs.smoothing_sigmas = [[8, 4, 2, 1, 0], [4, 3, 2, 1, 0], [2, 1, 0]]
        else:
            reg.inputs.number_of_iterations = ([[5000, 5000, 1000, 500],
                                                [5000, 5000, 1000, 500],
                                                [75, 50]])
            reg.inputs.shrink_factors = [[7, 5, 2, 1], [4, 3, 2, 1], [2, 1]]
            reg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [3, 2, 1, 0], [0.5, 0]]
        reg.inputs.convergence_threshold = [1.e-6] * 3
        reg.inputs.sigma_units = ['vox']*3
        reg.inputs.transform_parameters = [(0.25,),
                                           (0.25,),
                                           (0.25,)]
        reg.inputs.use_estimate_learning_rate_once = [True] * 3
        reg.inputs.use_histogram_matching = [False, False, True]
    elif reg_type == SYN:
        reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        reg.inputs.metric = ['MI', 'MI', ['MI', 'CC']]
        reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
        reg.inputs.radius_or_number_of_bins = [32, 32, [32, 4]]
        reg.inputs.convergence_window_size = [5, 5, 5]
        reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
        reg.inputs.sampling_percentage = [0.5] * 2 + [[None, None]]
        if reg.inputs.initial_moving_transform_com:
            reg.inputs.number_of_iterations = ([[10000, 10000, 1000, 1000, 1000],
                                                [10000, 10000, 1000, 1000, 1000],
                                                [100, 75, 75, 75]])
            reg.inputs.shrink_factors = [[9, 5, 3, 2, 1], [5, 4, 3, 2, 1], [5, 3, 2, 1]]
            reg.inputs.smoothing_sigmas = [[8, 4, 2, 1, 0], [4, 3, 2, 1, 0], [4, 2, 1, 0]]
        else:
            reg.inputs.number_of_iterations = ([[5000, 5000, 1000, 500],
                                                [5000, 5000, 1000, 500],
                                                [100, 90, 75]])
            reg.inputs.shrink_factors = [[7, 5, 2, 1], [4, 3, 2, 1], [4, 2, 1]]
            reg.inputs.smoothing_sigmas = [[6, 4, 1, 0], [3, 2, 1, 0], [1, 0.5, 0]]
        reg.inputs.convergence_threshold = [1.e-6] * 2 + [-0.01]
        reg.inputs.sigma_units = ['vox']*3
        reg.inputs.transform_parameters = [(0.25,),
                                           (0.25,),
                                           (0.2, 3.0, 0.0)]
        reg.inputs.use_estimate_learning_rate_once = [True] * 3
        reg.inputs.use_histogram_matching = [False, False, True]
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
    util.LOGGER.info("starting registration")
    start_time = datetime.datetime.now()
    util.LOGGER.info(reg.cmdline)
    reg.run()
    util.LOGGER.info("Finished registration: ")
    util.LOGGER.info(datetime.datetime.now() - start_time)

    util.transform_volume(moving_img.pre_processed_filepath, transform,
                          outputpath=moving_img.processed_filepath)
    shutil.copy(transform, moving_img.transform)
    util.generate_image(moving_img.processed_filepath, fixed)

    return moving_img


def process_dataset(args):
    """ pre process and registrate volume"""
    (moving_image_id, reg_type, save_to_db, be_method, reg_type_be) = args
    util.LOGGER.info(moving_image_id)

    for k in range(3):
        try:
            start_time = datetime.datetime.now()
            img = img_data(moving_image_id, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
            img = pre_process(img, reg_type=reg_type_be, be_method=be_method)
            util.LOGGER.info("-- Run time preprocess: ")
            util.LOGGER.info(datetime.datetime.now() - start_time)

            img = registration(img, util.TEMPLATE_MASKED_VOLUME, reg_type)
            break
        # pylint: disable= broad-except
        except Exception as exp:
            util.LOGGER.error('Crashed during ' + str(k+1) + ' of 3 ' + str(exp))
    util.LOGGER.info(" -- Run time: " + str(datetime.datetime.now() - start_time))
    if save_to_db:
        save_transform_to_database([img])
        del img


# pylint: disable= too-many-arguments
def get_transforms(moving_dataset_image_ids, reg_type=None,
                   process_dataset_func=process_dataset, save_to_db=False,
                   be_method=BE_METHOD, reg_type_be=None):
    """Calculate transforms """
    if not reg_type_be:
        reg_type_be = reg_type
    if MULTITHREAD > 1:
        if MULTITHREAD == 'max':
            pool = Pool()
        else:
            pool = Pool(MULTITHREAD)
        # http://stackoverflow.com/a/1408476/636384
        pool.map_async(process_dataset_func,
                       zip(moving_dataset_image_ids,
                           [reg_type]*len(moving_dataset_image_ids),
                           [save_to_db]*len(moving_dataset_image_ids),
                           [be_method]*len(moving_dataset_image_ids),
                           [reg_type_be]*len(moving_dataset_image_ids))).get(999999999)
        pool.close()
        pool.join()
    else:
        for moving_image_id in moving_dataset_image_ids:
            process_dataset_func((moving_image_id, reg_type, save_to_db, be_method))


def move_vol(moving, transform, label_img=False, slice_size=1, ref_img=None):
    """ Move data with transform """
    if label_img:
        # resample volume to 1 mm slices
        target_affine_3x3 = np.eye(3) * slice_size
        img_3d_affine = resample_img(moving, target_affine=target_affine_3x3,
                                     interpolation='nearest')
        resampled_file = util.TEMP_FOLDER_PATH + util.get_basename(moving) + '_resample.nii.gz'
        # pylint: disable= no-member
        img_3d_affine.to_filename(resampled_file)
        del img_3d_affine
    else:
        img = img_data(-1, util.DATA_FOLDER, util.TEMP_FOLDER_PATH)
        img.set_img_filepath(moving)
        resampled_file = pre_process(img, False).pre_processed_filepath

    result = util.transform_volume(resampled_file, transform, label_img, ref_img=ref_img)
    util.generate_image(result, util.TEMPLATE_VOLUME)
    return result


def save_transform_to_database(imgs):
    """ Save data transforms to database"""
    # pylint: disable= too-many-locals, bare-except
    conn = sqlite3.connect(util.DB_PATH, timeout=900)
    conn.text_factory = str

    try:
        conn.execute("alter table Images add column 'registration_date' 'TEXT'")
    except sqlite3.OperationalError:
        pass

    for img in imgs:
        cursor = conn.execute('''SELECT pid from Images where id = ? ''', (img.image_id,))
        pid = cursor.fetchone()[0]

        folder = util.DATA_FOLDER + str(pid) + "/registration_transforms/"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        transform_paths = ""
        util.LOGGER.info(img.get_transforms())
        for _transform in img.get_transforms():
            util.LOGGER.info(_transform)
            dst_file = folder + util.get_basename(_transform) + '.h5.gz'
            with open(_transform, 'rb') as f_in, gzip.open(dst_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            transform_paths += str(pid) + "/registration_transforms/" +\
                util.get_basename(_transform) + '.h5.gz' + ", "
        transform_paths = transform_paths[:-2]

        cursor2 = conn.execute('''UPDATE Images SET transform = ? WHERE id = ?''',
                               (transform_paths, img.image_id))
        cursor2 = conn.execute('''UPDATE Images SET fixed_image = ? WHERE id = ?''',
                               (img.fixed_image, img.image_id))

        cursor2 = conn.execute('''UPDATE Images SET registration_date = ? WHERE id = ?''',
                               (datetime.datetime.now().strftime("%Y-%m-%d"), img.image_id))

        folder = util.DATA_FOLDER + str(pid) + "/reg_volumes_labels/"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        vol_path = util.compress_vol(img.processed_filepath)
        shutil.copy(vol_path, folder)

        volume_db = str(pid) + "/reg_volumes_labels/" + basename(vol_path)
        cursor2 = conn.execute('''UPDATE Images SET filepath_reg = ? WHERE id = ?''',
                               (volume_db, img.image_id))

        cursor = conn.execute('''SELECT filepath, id from Labels where image_id = ? ''',
                              (img.image_id,))
        for (filepath, label_id) in cursor:
            temp = util.compress_vol(move_vol(util.DATA_FOLDER + filepath,
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
