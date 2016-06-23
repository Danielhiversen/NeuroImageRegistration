
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

To download data:
    ipython
        from nilearn import datasets
        data = datasets.fetch_icbm152_2009()
        TEMPLATE_VOLUME = data.get("t1")

"""
# pylint: disable= redefined-builtin
# import nipype.interfaces.dipy as dipy
from __future__ import print_function
from __future__ import division
import sys
from multiprocessing import Pool
import os
from os.path import basename
from os.path import splitext
import sqlite3
from builtins import map
from builtins import str
from builtins import range
from nilearn.image import resample_img
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nibabel as nib
import numpy as np

# from dipy.align.aniso2iso import resample

MULTITHREAD = 1  # 1,23,4....., "max"
MULTITHREAD = "max"

TEMP_FOLDER_PATH = ""
TEMPLATE_VOLUME = ""
DATA_FOLDER = ""
DB_PATH = ""
TEMPLATE_MASK = ""

DWICONVERT_PATH = ""
DATA_PATH_LISA = ""
PID_LISA = ""
DATA_PATH_LISA_QOL = ""
DATA_PATH_ANNE_LISE = ""
PID_ANNE_LISE = ""
DATA_PATH_LGG = ""

OUT_FOLDER = ""
DB_PATH = ""

RIGID = 'rigid'
AFFINE = 'affine'
SYN = 'syn'

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


def setup(temp_path):
    """setup for current computer """
    # pylint: disable= global-statement
    global TEMP_FOLDER_PATH
    TEMP_FOLDER_PATH = temp_path
    setup_paths()


def setup_paths():
    """setup for current computer """
    # pylint: disable= global-statement, line-too-long
    global TEMPLATE_VOLUME, TEMPLATE_MASK, DATA_FOLDER, DB_PATH
    global DWICONVERT_PATH, DATA_PATH_LISA, PID_LISA, DATA_PATH_LISA_QOL, DATA_PATH_ANNE_LISE, PID_ANNE_LISE, DATA_PATH_LGG
    hostname = os.uname()[1]
    if hostname == 'dahoiv-Alienware-15':
        TEMPLATE_VOLUME = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'

        DATA_FOLDER = "/mnt/dokumneter/data/database/"
        DB_PATH = DATA_FOLDER + "brainSegmentation.db"

        DWICONVERT_PATH = "/home/dahoiv/disk/kode/Slicer/Slicer-SuperBuild/Slicer-build/lib/Slicer-4.5/cli-modules/DWIConvert"

        main_folder = "/mnt/sintef/NevroData/Segmentations/"
        DATA_PATH_LISA = main_folder + "Segmenteringer_Lisa/"
        PID_LISA = main_folder + "Koblingsliste__Lisa.xlsx"
        DATA_PATH_LISA_QOL = main_folder + "Segmenteringer_Lisa/Med_QoL/"
        DATA_PATH_ANNE_LISE = main_folder + "Segmenteringer_AnneLine/"
        PID_ANNE_LISE = main_folder + "Koblingsliste__Anne_Line.xlsx"
        DATA_PATH_LGG = main_folder + "Data_HansKristian_LGG/LGG/NIFTI/"

    elif hostname == 'dahoiv-Precision-M6500':
        TEMPLATE_VOLUME = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

        DATA_FOLDER = "/mnt/dokumenter/daniel/database/"
        DATA_FOLDER = "/home/dahoiv/ingerid-pc/media/ingerid/data/daniel/database/"
        DB_PATH = DATA_FOLDER + "brainSegmentation.db"

        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        TEMPLATE_VOLUME = "/home/daniel/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/daniel/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/daniel/antsbin/bin/'

        DATA_FOLDER = "/media/ingerid/data/daniel/database/"
        DB_PATH = DATA_FOLDER + "brainSegmentation.db"

        DWICONVERT_PATH = "/home/daniel/Slicer-4.5.0-1-linux-amd64/lib/Slicer-4.5/cli-modules/DWIConvert"

        main_folder = "/home/daniel/Sintef/NevroData/Segmentations/"
        DATA_PATH_LISA = main_folder + "Segmenteringer_Lisa/"
        PID_LISA = main_folder + "Koblingsliste__Lisa.xlsx"
        DATA_PATH_LISA_QOL = main_folder + "Segmenteringer_Lisa/Med_QoL/"
        DATA_PATH_ANNE_LISE = main_folder + "Segmenteringer_AnneLine/"
        PID_ANNE_LISE = main_folder + "Koblingsliste__Anne_Line.xlsx"
        DATA_PATH_LGG = main_folder + "Data_HansKristian_LGG/LGG/NIFTI/"

    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " + sys.argv[0])
        raise Exception


def prepare_template(template_vol, template_mask):
    """ prepare template volumemoving"""
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = template_vol
    mult.inputs.second_input = template_mask
    mult.inputs.output_product_image = TEMP_FOLDER_PATH + "masked_template.nii"
    mult.run()


def pre_process(input_file, do_bet=True):
    """ Pre process the data"""
    n4_file = TEMP_FOLDER_PATH + splitext(splitext(basename(input_file))[0])[0]\
        + '_n4.nii'
    norm_file = TEMP_FOLDER_PATH + splitext(basename(n4_file))[0]\
        + '_norm.nii'
    resampled_file = TEMP_FOLDER_PATH + splitext(basename(norm_file))[0]\
        + '_resample.nii'
    out_file = TEMP_FOLDER_PATH +\
        splitext(basename(resampled_file))[0] +\
        '_bet.nii.gz'

    if os.path.exists(out_file):
        return out_file

    n4bias = ants.N4BiasFieldCorrection()
    n4bias.inputs.dimension = 3
    n4bias.inputs.input_image = input_file
    n4bias.inputs.output_image = n4_file
    n4bias.run()

    # normalization [0,100], same as template
    img = nib.load(n4_file)
    result_img = nib.Nifti1Image(img.get_data()/np.amax(img.get_data())*100, img.affine, img.header)
    result_img.to_filename(norm_file)

    # resample volume to 1 mm slices
    target_affine_3x3 = np.eye(3) * 1
    img_3d_affine = resample_img(norm_file, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    if not do_bet:
        return resampled_file

    # brain_extraction(resampled_file, out_file)
    # return out_file

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
    bet.inputs.out_file = out_file

    bet.run()
    print(out_file)

    # generate_image(out_file, resampled_file)

    return out_file


# Not used for the moment
def brain_extraction(in_file, out_file):
    """ Brain extraction."""
    reg = ants.Registration()
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = TEMPLATE_VOLUME
    reg.inputs.moving_image = in_file
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads = 1
    reg.inputs.transforms = ['Rigid']
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.convergence_window_size = [10]
    reg.inputs.metric = ['CC']
    reg.inputs.metric_weight = [1.0]*3
    reg.inputs.number_of_iterations = [[1000, 500, 250, 100]]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Regular']
    reg.inputs.sampling_percentage = [0.25]
    reg.inputs.shrink_factors = [[8, 4, 2, 1]]
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.transform_parameters = [(0.1,)]
    reg.inputs.use_histogram_matching = True
    reg.inputs.write_composite_transform = True

    name = splitext(splitext(basename(in_file))[0])[0]
    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + "output_"+name+'_'
    reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '_regRigid.nii'
    reg.run()

    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = TEMP_FOLDER_PATH + name + '_regRigid.nii'
    mult.inputs.second_input = TEMPLATE_MASK
    mult.inputs.output_product_image = out_file
    mult.run()


def registration(moving, fixed, reg_type):
    """Image2Image registration """
    reg = ants.Registration()
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads = 1
    if reg_type == RIGID:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.sampling_strategy = [None]
        reg.inputs.sampling_percentage = [1]
        reg.inputs.metric = ['CC']
        reg.inputs.radius_or_number_of_bins = [4]
    elif reg_type == AFFINE:
        reg.inputs.transforms = ['Rigid', 'Affine']
        reg.inputs.sampling_strategy = ['Regular', None]
        reg.inputs.sampling_percentage = [0.25, 1]
        reg.inputs.metric = ['MI', 'CC']
        reg.inputs.radius_or_number_of_bins = [32, 4]
    elif reg_type == SYN:
        reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        reg.inputs.sampling_strategy = ['Regular', 'Regular', None]
        reg.inputs.sampling_percentage = [0.25, 0.25, 1]
        reg.inputs.metric = ['MI', 'MI', 'CC']
        reg.inputs.radius_or_number_of_bins = [32, 32, 4]
    else:
        raise Exception("Wrong registration format " + reg_type)
    reg.inputs.metric_weight = [1.0]*3
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.convergence_window_size = [10]
    reg.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                       [1000, 500, 250, 100],
                                       [100, 70, 50, 20]]
    reg.inputs.shrink_factors = [[8, 4, 2, 1]]*3
    reg.inputs.smoothing_sigmas = [[8, 4, 1, 0]]*3
    reg.inputs.sigma_units = ['vox']*3
    reg.inputs.transform_parameters = [(0.1,),
                                       (0.1,),
                                       (0.2, 3.0, 0.0)]
    reg.inputs.use_histogram_matching = True
    reg.inputs.write_composite_transform = True

    name = splitext(splitext(basename(moving))[0])[0] + '_' + reg_type
    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + name
    reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '.nii'

    result = TEMP_FOLDER_PATH + name + 'Composite.h5'
    print(result)
    if os.path.exists(result):
        # generate_image(reg.inputs.output_warped_image, fixed)
        return result
    reg.run()
    # generate_image(reg.inputs.output_warped_image, fixed)

    return result


def process_dataset(args, num_tries=3):
    """ pre process and registrate volume"""
    (moving_image_id, reg_type) = args
    print(moving_image_id, reg_type)
    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cursor = conn.execute('''SELECT filepath from Images where id = ? ''', (moving_image_id,))
    moving = DATA_FOLDER + cursor.fetchone()[0]
    cursor.close()
    conn.close()

    for k in range(num_tries):
        try:
            moving_pre_processed = pre_process(moving)
            transform = registration(moving_pre_processed,
                                     TEMP_FOLDER_PATH + "masked_template.nii",
                                     reg_type)
            return (moving_image_id, transform, -1)
        # pylint: disable=  broad-except
        except Exception as exp:
            print(exp)
            raise Exception('Crashed during processing of ' + moving + '. Try ' +
                            str(k+1) + ' of ' + str(num_tries) + ' \n' + str(exp))


def get_transforms(moving_dataset_image_ids, reg_type=None, process_dataset_func=process_dataset):
    """Calculate transforms """
    print(moving_dataset_image_ids)
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
        cursor = conn.execute('''SELECT transform, fixed_image from Images where id = ? ''', (_id,))
        db_temp = cursor.fetchone()
        img = DATA_FOLDER + db_temp[0]

        temp = move_vol(img, transforms)
        label = "img"
        if label in result:
            result[label].append(temp)
        else:
            result[label] = [temp]

        for (segmentation, label) in find_seg_images(_id):
            temp = move_vol(segmentation, transforms)
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


def move_vol(moving, transform):
    """ Move data with transform """

    resampled_file = pre_process(moving, False)
    result = TEMP_FOLDER_PATH + splitext(basename(resampled_file))[0] + '_reg.nii'
#    if os.path.exists(result):
#        return result

    apply_transforms = ants.ApplyTransforms()
    apply_transforms.inputs.dimension = 3
    apply_transforms.inputs.input_image = resampled_file
    apply_transforms.inputs.reference_image = TEMPLATE_VOLUME
    apply_transforms.inputs.output_image = result
    apply_transforms.inputs.interpolation = 'NearestNeighbor'
    apply_transforms.inputs.default_value = 0
    apply_transforms.inputs.transforms = transform
    apply_transforms.inputs.invert_transform_flags = [False]*len(transform)
    print(apply_transforms.cmdline)
    apply_transforms.run()

    print(apply_transforms.inputs.output_image, result)

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
