
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

"""
# pylint: disable= redefined-builtin
# import nipype.interfaces.dipy as dipy
from __future__ import print_function
from __future__ import division
import sys
import errno
from multiprocessing import Pool
import os
from os.path import basename
from os.path import splitext
import sqlite3
import shutil
from builtins import map
from builtins import str
from nilearn import datasets
from nilearn.image import resample_img
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nibabel as nib
import numpy as np

from img_data import img_data

# from dipy.align.aniso2iso import resample

MULTITHREAD = 1  # 1,23,4....., "max"
#MULTITHREAD = "max"

TEMPLATE_VOLUME = datasets.fetch_icbm152_2009(data_dir="./").get("t1")
TEMPLATE_MASK = datasets.fetch_icbm152_2009(data_dir="./").get("mask")

TEMP_FOLDER_PATH = ""
DATA_FOLDER = ""
DB_PATH = ""

RIGID = 'rigid'
AFFINE = 'affine'
SYN = 'syn'


BET_METHOD = 0

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


def setup(temp_path, datatype):
    """setup for current computer """
    # pylint: disable= global-statement
    global TEMP_FOLDER_PATH
    TEMP_FOLDER_PATH = temp_path
    setup_paths(datatype)


def setup_paths(datatype):
    """setup for current computer """
    # pylint: disable= global-statement, line-too-long
    if datatype not in ["LGG", "GBM"]:
        print("Unkown datatype " + datatype)
        raise Exception

    global DATA_FOLDER, DB_PATH

    hostname = os.uname()[1]
    if hostname == 'dahoiv-Alienware-15':
        DATA_FOLDER = "/mnt/dokumneter/data/database/"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'
    elif hostname == 'dahoiv-Precision-M6500':
        DATA_FOLDER = "/home/dahoiv/database/"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        DATA_FOLDER = "/media/ingerid/data/daniel/database/"
        os.environ["PATH"] += os.pathsep + '/home/daniel/antsbin/bin/'
    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " + sys.argv[0])
        raise Exception

    DATA_FOLDER = DATA_FOLDER + datatype + "/"
    DB_PATH = DATA_FOLDER + "brainSegmentation.db"


def prepare_template(template_vol, template_mask):
    """ prepare template volumemoving"""
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = template_vol
    mult.inputs.second_input = template_mask
    mult.inputs.output_product_image = TEMP_FOLDER_PATH + "masked_template.nii"
    mult.run()


def pre_process(img, do_bet=True):
    # pylint: disable= too-many-statements
    """ Pre process the data"""

    input_file = img.img_filepath
    n4_file = TEMP_FOLDER_PATH + splitext(splitext(basename(input_file))[0])[0]\
        + '_n4.nii'
    norm_file = TEMP_FOLDER_PATH + splitext(basename(n4_file))[0]\
        + '_norm.nii'
    resampled_file = TEMP_FOLDER_PATH + splitext(basename(norm_file))[0]\
        + '_resample.nii'
    img.pre_processed_filepath = TEMP_FOLDER_PATH +\
        splitext(basename(resampled_file))[0] +\
        '_bet.nii.gz'

    n4bias = ants.N4BiasFieldCorrection()
    n4bias.inputs.dimension = 3
    n4bias.inputs.input_image = input_file
    n4bias.inputs.output_image = n4_file
    n4bias.run()

    # normalization [0,100], same as template
    normalize_img = nib.load(n4_file)
    result_img = nib.Nifti1Image(normalize_img.get_data()/np.amax(normalize_img.get_data())*100,
                                 normalize_img.affine, normalize_img.header)
    result_img.to_filename(norm_file)

    # resample volume to 1 mm slices
    target_affine_3x3 = np.eye(3) * 1
    img_3d_affine = resample_img(norm_file, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    if not do_bet:
        img.pre_processed_filepath = resampled_file
        return img

    if BET_METHOD == 0:
        print("Doing registration for bet")
        name = splitext(splitext(basename(resampled_file))[0])[0] + "_bet"
        reg = ants.Registration()
        # reg.inputs.args = "--verbose 1"
        reg.inputs.collapse_output_transforms = True
        reg.inputs.fixed_image = TEMPLATE_VOLUME
        reg.inputs.moving_image = resampled_file
        reg.inputs.initial_moving_transform_com = True
        reg.inputs.num_threads = 8
        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.sampling_strategy = ['Regular'] * 2 
  
        reg.inputs.sampling_percentage = [0.3] * 2
        reg.inputs.metric = ['Mattes'] * 2
        reg.inputs.radius_or_number_of_bins = [32] * 2 
        reg.inputs.metric_weight = [1] * 2
        reg.inputs.winsorize_lower_quantile = 0.005
        reg.inputs.winsorize_upper_quantile = 0.995
        reg.inputs.convergence_window_size = [20] * 2 
        reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2)
        reg.inputs.convergence_threshold = [1.e-8] * 2
        reg.inputs.shrink_factors = [[3, 2, 1]]*2 
        reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2
        reg.inputs.sigma_units = ['vox']*3
        reg.inputs.transform_parameters = [(0.1,), (0.1,)]
        reg.inputs.use_histogram_matching = [False] * 2 
        reg.inputs.use_estimate_learning_rate_once = [True] * 2
    
        reg.inputs.write_composite_transform = True
        # reg.inputs.fixed_image_mask = img.label_inv_filepath

        reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + name
        reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '_betReg.nii'
        print("starting bet registration")
        reg.run()
        print("Finished bet registration")

        img.init_transform = TEMP_FOLDER_PATH + name + 'Composite.h5'

        mult = ants.MultiplyImages()
        mult.inputs.dimension = 3
        mult.inputs.first_input = reg.inputs.output_warped_image
        mult.inputs.second_input = TEMPLATE_MASK
        mult.inputs.output_product_image = img.pre_processed_filepath
        mult.run()

    elif BET_METHOD == 1:
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
        bet.inputs.reduce_bias = True
        bet.inputs.out_file = img.pre_processed_filepath

        bet.run()
    print("---BET", img.pre_processed_filepath)
    return img


def registration(moving_img, fixed, reg_type):
    # pylint: disable= too-many-statements
    """Image2Image registration """
    name = splitext(splitext(basename(moving_img.pre_processed_filepath))[0])[0] + '_' + reg_type

    reg = ants.Registration()
    # reg.inputs.args = "--verbose 1"
    reg.inputs.collapse_output_transforms = True
    reg.inputs.moving_image = moving_img.pre_processed_filepath
    reg.inputs.fixed_image = fixed
    init_moving_transform = moving_img.init_transform
    if init_moving_transform is not None and os.path.exists(init_moving_transform):
        reg.inputs.initial_moving_transform = init_moving_transform
        print("Found initial transform")
    else:
        reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads = 8
    if reg_type == RIGID:
        reg.inputs.transforms = ['Rigid', 'Rigid']
        reg.inputs.sampling_strategy = ['Regular', None]
        reg.inputs.sampling_percentage = [0.25, 1]
        reg.inputs.metric = ['MI', 'CC']
        reg.inputs.radius_or_number_of_bins = [32, 4]
    elif reg_type == AFFINE:
        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.sampling_strategy = ['Regular', None]
        reg.inputs.sampling_percentage = [0.25, 1]
        reg.inputs.metric = ['MI', 'CC']
        reg.inputs.radius_or_number_of_bins = [32, 4]
    elif reg_type == SYN:
        reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
        reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
        reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
        reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    else:
        raise Exception("Wrong registration type " + reg_type)
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2 + [[100, 50, 30]])
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox']*3
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.collapse_output_transforms = True
#    reg.inputs.fixed_image_mask = moving_img.label_inv_filepath
    reg.inputs.write_composite_transform = True

    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + name
    reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '.nii'

    result = TEMP_FOLDER_PATH + name + 'Composite.h5'
    moving_img.transform = result
    print(result)
    if os.path.exists(result):
        #        generate_image(reg.inputs.output_warped_image, fixed)
        return moving_img
    reg.run()
    #    generate_image(reg.inputs.output_warped_image, fixed)

    return moving_img


def process_dataset(args):
    """ pre process and registrate volume"""
    (moving_image_id, reg_type) = args
    print(moving_image_id, reg_type)

    import datetime
    now =  datetime.datetime.now()
    img = img_data(moving_image_id, DATA_FOLDER, TEMP_FOLDER_PATH)
    img = pre_process(img)
    img = registration(img,
                       TEMP_FOLDER_PATH + "masked_template.nii",
                       reg_type)
    print("\n\n\n\n -- Run time: ")
    print(now - datetime.datetime.now())
    return (img, -1)


def get_transforms(moving_dataset_image_ids, reg_type=None, process_dataset_func=process_dataset):
    """Calculate transforms """
    if MULTITHREAD > 1:
        if MULTITHREAD == 'max':
            pool = Pool()
        else:
            pool = Pool(MULTITHREAD)
        # http://stackoverflow.com/a/1408476/636384
        result = pool.map_async(process_dataset_func,
                                zip(moving_dataset_image_ids,
                                    [reg_type]*len(moving_dataset_image_ids))).get(999999999)
        pool.close()
        pool.join()
    else:
        result = list(map(process_dataset_func, zip(moving_dataset_image_ids,
                                                    [reg_type]*len(moving_dataset_image_ids))))
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


def post_calculations(moving_dataset_image_ids):
    """ Transform images and calculate avg"""
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    result = dict()
    for _id in moving_dataset_image_ids:
        transforms = get_transforms_from_db(_id, conn)
        cursor = conn.execute('''SELECT filepath from Images where id = ? ''', (_id,))
        db_temp = cursor.fetchone()
        img = DATA_FOLDER + db_temp[0]
        temp = move_vol(img, transforms)
        print(img)
        label = "img"
        if label in result:
            result[label].append(temp)
        else:
            result[label] = [temp]

        for (segmentation, label) in find_seg_images(_id):
            temp = move_vol(segmentation, transforms, True)
            if label in result:
                result[label].append(temp)
            else:
                result[label] = [temp]

    cursor.close()
    conn.close()

    for label in result.keys():
        avg_calculation(result[label], label)


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


def move_vol(moving, transform, label_img=False):
    """ Move data with transform """
    apply_transforms = ants.ApplyTransforms()

    if label_img:
        # resample volume to 1 mm slices
        target_affine_3x3 = np.eye(3) * 1
        img_3d_affine = resample_img(moving, target_affine=target_affine_3x3,
                                     interpolation='nearest')
        resampled_file = TEMP_FOLDER_PATH + splitext(basename(moving))[0] + '_resample.nii'
        nib.save(img_3d_affine, resampled_file)
        apply_transforms.inputs.interpolation = 'NearestNeighbor'
    else:
        img = img_data(-1, DATA_FOLDER, TEMP_FOLDER_PATH)
        img.set_img_filepath(moving)
        resampled_file = pre_process(img, False).img_filepath
        apply_transforms.inputs.interpolation = 'Linear'

    result = TEMP_FOLDER_PATH + splitext(basename(resampled_file))[0] + '_reg.nii'
#    if os.path.exists(result):
#        return result

    apply_transforms.inputs.dimension = 3
    apply_transforms.inputs.input_image = resampled_file
    apply_transforms.inputs.reference_image = TEMPLATE_VOLUME
    apply_transforms.inputs.output_image = result
    apply_transforms.inputs.default_value = 0
    apply_transforms.inputs.transforms = transform
    apply_transforms.inputs.invert_transform_flags = [False]*len(transform)
    apply_transforms.run()

    generate_image(apply_transforms.inputs.output_image, TEMPLATE_VOLUME)

    return apply_transforms.inputs.output_image


def avg_calculation(images, label):
    """ Calculate average volumes """
    path = TEMP_FOLDER_PATH + 'avg_' + label + '.nii'
    path = path.replace('label', 'tumor')

    average = None
    for file_name in images:
        img = nib.load(file_name)
        if average is None:
            average = np.zeros(img.get_data().shape)
        average = average + np.array(img.get_data())
    average = average / float(len(images))
    result_img = nib.Nifti1Image(average, img.affine)
    result_img.to_filename(path)

    generate_image(path, TEMPLATE_VOLUME)


def generate_image(path, path2):
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
    name = splitext(splitext(basename(path))[0])[0]
    plt.suptitle(name)

    plt.savefig(splitext(splitext(path)[0])[0] + ".png")
    plt.close()


def save_transform_to_database(data_transforms):
    """ Save data transforms to database"""
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    for img, fixed_imag_id in data_transforms:
        cursor = conn.execute('''SELECT pid from Images where id = ? ''', (img.image_id,))
        pid = cursor.fetchone()[0]

        folder = DATA_FOLDER + str(pid) + "/registration_transforms/"
        mkdir_p(folder)

        transform_paths = ""
        print(img.get_transforms())
        for _transform in img.get_transforms():
            print(_transform)
            dst_file = folder + basename(_transform)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(_transform, folder)
            transform_paths += str(pid) + "/registration_transforms/" + basename(_transform) + ", "
        transform_paths = transform_paths[:-2]

        cursor2 = conn.execute('''UPDATE Images SET transform = ? WHERE id = ?''',
                               (transform_paths, img.image_id))
        cursor2 = conn.execute('''UPDATE Images SET fixed_image = ? WHERE id = ?''',
                               (fixed_imag_id, img.image_id))

        conn.commit()
        cursor.close()
        cursor2.close()

    cursor = conn.execute('''VACUUM; ''')
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
