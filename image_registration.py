
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
import glob
import sys
from multiprocessing import Pool
import os
from os.path import basename
from os.path import splitext
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

DATA_PATH = ""
T1_PATTERN = []
DATA_OUT_PATH = ""
TEMP_FOLDER_PATH = ""
TEMPLATE_VOLUME = ""
TEMPLATE_MASK = ""
DEFORMATION = False

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


def setup(argv):
    # pylint: disable= too-many-branches, global-statement, line-too-long, too-many-statements
    """setup for current computer """
    global DATA_PATH, T1_PATTERN, DATA_OUT_PATH, TEMP_FOLDER_PATH, TEMPLATE_VOLUME, TEMPLATE_MASK, DEFORMATION
    hostname = os.uname()[1]

    dataset = argv[0]

    if dataset == "HGG":
        T1_PATTERN = ['T1_diag', 'T1_preop']
        data_folder = 'out_HGG/'
    elif dataset == "LGG_PRE":
        T1_PATTERN = ['_pre.nii']
        data_folder = 'out_LGG_PRE/'
    elif dataset == "LGG_POST":
        T1_PATTERN = ['_post.nii']
        data_folder = 'out_LGG_POST/'
    else:
        print("Unkown dataset: ", dataset)
        raise Exception

    if len(argv) > 1 and argv[1] == "DEF":
        DEFORMATION = True
        data_folder = data_folder[:-1] + "_def/"
    else:
        DEFORMATION = False

    TEMP_FOLDER_PATH = "temp_" + data_folder

    if hostname == 'dahoiv-Alienware-15':
        if dataset == "HGG":
            DATA_PATH = '/home/dahoiv/disk/data/tumor_segmentation/'
        elif dataset == "LGG_PRE":
            DATA_PATH = '/home/dahoiv/disk/data/LGG_kart/PRE/'
        elif dataset == "LGG_POST":
            DATA_PATH = '/home/dahoiv/disk/data/LGG_kart/POST/'
        else:
            print("Unkown dataset")
            raise Exception
        DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/' + data_folder
        TEMPLATE_VOLUME = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'
    elif hostname == 'dahoiv-Precision-M6500':
        if dataset == "HGG":
            DATA_PATH = '/mnt/dokumenter/data/tumor_segmentation/'
        elif dataset == "LGG_PRE":
            DATA_PATH = '/mnt/dokumenter/data/LGG_kart/PRE/'
        elif dataset == "LGG_POST":
            DATA_PATH = '/mnt/dokumenter/data/LGG_kart/POST/'
        else:
            print("Unkown dataset")
            raise Exception

        DATA_OUT_PATH = '/mnt/dokumenter/NeuroImageRegistration/' + data_folder
        TEMPLATE_VOLUME = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    elif hostname == 'ingerid-PC':
        if dataset == "HGG":
            DATA_PATH = '/home/daniel/data/HGG/'
        elif dataset == "LGG_PRE":
            DATA_PATH = '/home/daniel/data/LGG_PRE/'
        elif dataset == "LGG_POST":
            DATA_PATH = '/home/daniel/data/LGG_POST/'
        else:
            print("Unkown dataset")
            raise Exception

        DATA_OUT_PATH = '/home/daniel/data_out' + data_folder
        TEMPLATE_VOLUME = "/home/daniel/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/daniel/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/daniel/antsbin/bin/'

    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " + sys.argv[0])
        raise Exception

    print("Setup: \n\n Datapath: " + DATA_PATH + "\n Data out: " + DATA_OUT_PATH + "\n Deformation in registration: " + str(DEFORMATION) + "\n\n")


def prepare_template():
    """ prepare template volumemoving"""
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = TEMPLATE_VOLUME
    mult.inputs.second_input = TEMPLATE_MASK
    mult.inputs.output_product_image = TEMP_FOLDER_PATH + "masked_template.nii"
    mult.run()


def pre_process(data):
    """ Pre process the data"""
    n4_file = TEMP_FOLDER_PATH + splitext(basename(data))[0]\
        + '_n4.nii'
    resampled_file = TEMP_FOLDER_PATH + splitext(basename(n4_file))[0]\
        + '_resample.nii'
    out_file = TEMP_FOLDER_PATH +\
        splitext(basename(resampled_file))[0] +\
        '_bet.nii.gz'

    n4bias = ants.N4BiasFieldCorrection()
    n4bias.inputs.dimension = 3
    n4bias.inputs.input_image = data
    n4bias.inputs.output_image = n4_file
    n4bias.run()

    target_affine_3x3 = np.eye(3) * 1  # 1 mm slices
    img_3d_affine = resample_img(data, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    # pylint: disable= using-constant-test
    if False:
        brain_extraction(resampled_file, out_file)
        return

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

    generate_image(out_file, resampled_file)

    return bet.inputs.out_file


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
    reg.inputs.metric = ['MI', 'MI', 'CC']
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
    generate_image(reg.inputs.output_warped_image)

    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = TEMP_FOLDER_PATH + name + '_regRigid.nii'
    mult.inputs.second_input = TEMPLATE_MASK
    mult.inputs.output_product_image = out_file
    mult.run()


def registration(moving, fixed):
    """Image2Image registration """
    reg = ants.Registration()
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads = 1
    if DEFORMATION:
        reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    else:
        reg.inputs.transforms = ['Rigid', 'Affine']
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.convergence_window_size = [10]
    reg.inputs.metric = ['MI', 'MI', 'CC']
    reg.inputs.metric_weight = [1.0]*3
    reg.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                       [1000, 500, 250, 100],
                                       [100, 70, 50, 20]]
    reg.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg.inputs.sampling_strategy = ['Regular', 'Regular', None]
    reg.inputs.sampling_percentage = [0.25, 0.25, 1]
    reg.inputs.shrink_factors = [[8, 4, 2, 1]]*3
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]*3
    reg.inputs.sigma_units = ['vox']*3
    reg.inputs.transform_parameters = [(0.1,),
                                       (0.1,),
                                       (0.2, 3.0, 0.0)]
    reg.inputs.use_histogram_matching = True
    reg.inputs.write_composite_transform = True

    name = splitext(splitext(basename(moving))[0])[0]
    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + "output_"+name+'_'
    reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '_reg.nii'

    result = reg.inputs.output_transform_prefix + 'Composite.h5'
    if os.path.exists(result):
        return result
    reg.run()
    generate_image(reg.inputs.output_warped_image)

    return result


def move_data(moving, transform):
    """ Move data with transform """
    resampled_file = TEMP_FOLDER_PATH + splitext(basename(moving))[0]\
        + '_resample.nii'
    result = DATA_OUT_PATH + splitext(basename(resampled_file))[0] + '_reg.nii'
    if os.path.exists(result):
        return result

    target_affine_3x3 = np.eye(3) * 1  # 1 mm slices
    img_3d_affine = resample_img(moving, target_affine=target_affine_3x3)
    nib.save(img_3d_affine, resampled_file)

    apply_transforms = ants.ApplyTransforms()
    apply_transforms.inputs.dimension = 3
    apply_transforms.inputs.input_image = resampled_file
    apply_transforms.inputs.reference_image = TEMPLATE_VOLUME
    apply_transforms.inputs.output_image = result
    apply_transforms.inputs.interpolation = 'NearestNeighbor'
    apply_transforms.inputs.default_value = 0
    apply_transforms.inputs.transforms = [transform]
    apply_transforms.inputs.invert_transform_flags = [False]
    # print(apply_transforms.cmdline)
    apply_transforms.run()

    generate_image(apply_transforms.inputs.output_image)

    return apply_transforms.inputs.output_image


def post_calculation(images, label):
    """ Calculate average volumes """
    path = DATA_OUT_PATH + 'avg_' + label + '.nii'
    path = path.replace('label', 'tumor')

#    avg = ants.AverageImages()
#    avg.inputs.dimension = 3
#    avg.inputs.output_average_image = path
#    avg.inputs.normalize = True
#    avg.inputs.images = images
#    print(avg.cmdline)
#    avg.run()

    average = None
    for file_name in images:
        img = nib.load(file_name)
        if average is None:
            average = np.zeros(img.get_data().shape)
        average = average + np.array(img.get_data())
    average = average
    result_img = nib.Nifti1Image(average, img.affine)
    result_img.to_filename(path)
    print(np.amax(average))
    print(np.amin(average))
    generate_image(path)


def find_moving_images():
    """ Find T1 image for registration """
    result = []
    for pattern in T1_PATTERN:
        result.extend(glob.glob(DATA_PATH + '*' + pattern + '*'))
    return result


def find_seg_images(moving):
    """ find corresponding images"""
    pattern = ''

    for char in basename(moving)[1:]:
        if char == '-':
            break
        pattern += str(char)
    result = glob.glob(os.path.dirname(moving) + '/k' + pattern + '*.nii')
    if len(result) == 0:  # LGG
        pattern = os.path.splitext(os.path.basename(moving))[0]
        result = glob.glob(os.path.dirname(moving) + '/'+pattern + '*.nii')
#    result .remove(moving)
    return result


def find_label(path):
    """Find label in file path """
    label = splitext(basename(path))[0]
    label = '_'.join(label.split("_")[1:])
    return label


def process_dataset(moving):
    """ pre process and registrate volume"""
    print(moving)
    num_tries = 3
    for k in range(num_tries):
        try:
            moving_pre_processed = pre_process(moving)
            transform = registration(moving_pre_processed,
                                     TEMP_FOLDER_PATH + "masked_template.nii")
            return (moving, transform)
        # pylint: disable=  broad-except
        except Exception as exp:
            print('Crashed during processing of ' + moving + '. Try ' +
                  str(k+1) + ' of ' + str(num_tries) + ' \n' + str(exp))


def move_dataset(moving_dataset):
    """ move dataset """
    if MULTITHREAD > 1:
        if MULTITHREAD == 'max':
            pool = Pool()
        else:
            pool = Pool(MULTITHREAD)
        # http://stackoverflow.com/a/1408476/636384
        result = pool.map_async(process_dataset, moving_dataset).get(999999999)
        pool.close()
        pool.join()
    else:
        result = list(map(process_dataset, moving_dataset))
    return result


def move_segmentations(transforms):
    """ move label image with transforms """
    result = dict()
    for moving, transform in transforms:
        for segmentation in find_seg_images(moving):
            print("         ", segmentation, transform)
            temp = move_data(segmentation, transform)
            label = find_label(temp)
            if label in result:
                result[label].append(temp)
            else:
                result[label] = [temp]
    return result


def generate_image(path, path2=TEMPLATE_VOLUME):
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

# pylint: disable= invalid-name
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python " + __file__ + " dataset")
        exit(1)
    os.nice(19)
    setup(sys.argv[1:])
    if not os.path.exists(TEMP_FOLDER_PATH):
        os.makedirs(TEMP_FOLDER_PATH)
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    prepare_template()

    moving_datasets = find_moving_images()
    data_transforms = move_dataset(moving_datasets)
    results = move_segmentations(data_transforms)

    for label_i in results:
        post_calculation(results[label_i], label_i)

    if sys.argv[1] == "test":
        print(len(results))
        print(len(moving_datasets))
        print(len(data_transforms))
        print(len(results))
