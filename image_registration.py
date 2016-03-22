""" install
wget -O- http://neuro.debian.net/lists/wily.de-m.full
sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkp://pgp.mit.edu:80 0xA5D32F012649A5A9
sudo apt-get update

sudo apt-get install libblas-dev liblapack-dev libfreetype6-dev
sudo apt-get install libpng16-dev fsl-complete cmake ninja-build
pip install --upgrade setuptools
pip install --upgrade distribute
sudo apt-get install python-pip matplotlib
sudo pip install dipy nipype

git clone git://github.com/stnava/ANTs.git
mkdir antsbin
cd antsbin
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ../ANTs/
ninja

"""
from __future__ import print_function
from __future__ import division
# pylint: disable= redefined-builtin
from builtins import map
from builtins import str
from builtins import range

# from dipy.align.aniso2iso import resample

import nipype.interfaces.ants as ants
# import nipype.interfaces.dipy as dipy
import nipype.interfaces.fsl as fsl

from os.path import basename
from os.path import splitext
import os
import sys
import glob
from multiprocessing import Pool

import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MULTITHREAD = 4  # 1,23,4....., "max"
MULTITHREAD = "max"

DATA_PATH = ""
T1_PATTERN = []
DATA_OUT_PATH = ""
TEMP_FOLDER_PATH = ""
TEMPLATE_VOLUME = ""
TEMPLATE_MASK = ""

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


def setup(dataset):
    # pylint: disable= too-many-branches, global-statement
    # pylint: disable= line-too-long
    """setup for current computer """
    global DATA_PATH, T1_PATTERN, DATA_OUT_PATH, TEMP_FOLDER_PATH, TEMPLATE_VOLUME, TEMPLATE_MASK
    if dataset == "HGG":
        T1_PATTERN = ['T1_diag', 'T1_preop']
        TEMP_FOLDER_PATH = 'temp_HGG/'
    elif dataset == "LGG":
        T1_PATTERN = ['_pre.nii']
        TEMP_FOLDER_PATH = 'temp_LGG/'
    elif dataset == "LGG_POST":
        T1_PATTERN = ['_post.nii']
        TEMP_FOLDER_PATH = 'temp_LGG_POST/'
    else:
        print("Unkown dataset")
        raise Exception

    hostname = os.uname()[1]
    if hostname == 'dahoiv-Alienware-15':
        if dataset == "HGG":
            DATA_PATH = '/home/dahoiv/disk/data/tumor_segmentation/'
            DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/out_HGG/'
        elif dataset == "LGG":
            DATA_PATH = '/home/dahoiv/disk/data/LGG_kart/PRE/'
            DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/out_LGG/'
        else:
            print("Unkown dataset")
            raise Exception
        TEMPLATE_VOLUME = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/'  # path to ANTs bin folder
    elif hostname == 'dahoiv-Precision-M6500':
        if dataset == "HGG":
            DATA_PATH = '/mnt/dokumenter/data/tumor_segmentation/'
            DATA_OUT_PATH = '/mnt/dokumenter/NeuroImageRegistration/out_HGG/'
        elif dataset == "LGG":
            DATA_PATH = '/mnt/dokumenter/data/LGG_kart/PRE/'
            DATA_OUT_PATH = '/mnt/dokumenter/NeuroImageRegistration/out_LGG/'
        elif dataset == "LGG_POST":
            DATA_PATH = '/mnt/dokumenter/data/LGG_kart/POST/'
            DATA_OUT_PATH = '/mnt/dokumenter/NeuroImageRegistration/out_LGG_POST/'
        else:
            print("Unkown dataset")
            raise Exception
        TEMPLATE_VOLUME = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/mnt/dokumenter/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        # path to ANTs bin folder
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/antsbin/bin/'
    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " + sys.argv[0])
        raise Exception


def prepare_template():
    """ prepare template volume"""
    mult = ants.MultiplyImages()
    mult.inputs.dimension = 3
    mult.inputs.first_input = TEMPLATE_VOLUME
    mult.inputs.second_input = TEMPLATE_MASK
    mult.inputs.output_product_image = TEMP_FOLDER_PATH + "masked_template.nii"
    mult.run()


def pre_process(data):
    """ Pre process the data"""
#    reslice = dipy.Resample()
#    reslice.inputs.in_file = data
#    reslice.inputs.out_file =
#    TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_temp.nii.gz'
#    reslice.run()

    # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
    bet = fsl.BET(command="fsl5.0-bet")
    bet.inputs.in_file = data
    # pylint: disable= pointless-string-statement
    """ fractional intensity threshold (0->1); default=0.5;
    smaller values give larger brain outline estimates"""
    bet.inputs.frac = 0.45
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
    bet.inputs.out_file = [TEMP_FOLDER_PATH +
                           splitext(basename(data))[0] +
                           '_bet.nii.gz']
    if os.path.exists(bet.inputs.out_file):
        return bet.inputs.out_file
    bet.run()
    print(TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_bet.nii.gz')

    return bet.inputs.out_file


def registration(moving, fixed):
    """Image2Image registration """
    reg = ants.Registration()
    reg.inputs.collapse_output_transforms = True
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads = 1
    reg.inputs.transforms = ['Rigid', 'Affine']  # , 'SyN']
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
    apply_transforms = ants.ApplyTransforms()
    apply_transforms.inputs.dimension = 3
    apply_transforms.inputs.input_image = moving
    apply_transforms.inputs.reference_image = TEMPLATE_VOLUME
    apply_transforms.inputs.output_image = [DATA_OUT_PATH +
                                            splitext(basename(moving))[0] +
                                            '_reg.nii']
    apply_transforms.inputs.interpolation = 'NearestNeighbor'
    apply_transforms.inputs.default_value = 0
    apply_transforms.inputs.transforms = [transform]
    apply_transforms.inputs.invert_transform_flags = [False]
    # print(apply_transforms.cmdline)
    apply_transforms.run()

    return apply_transforms.inputs.output_image


def post_calculation(images, label):
    """ Calculate average volumes """
    avg = ants.AverageImages()
    avg.inputs.dimension = 3
    avg.inputs.output_average_image = DATA_OUT_PATH + 'avg_' + label + '.nii'
    avg.inputs.normalize = True
    avg.inputs.images = images
    print(avg.cmdline)
    avg.run()
    generate_image(avg.inputs.output_average_image)


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


def generate_image(path):
    """ generate png images"""
    img = nib.load(path).get_data()
    img_template = nib.load(TEMPLATE_VOLUME).get_data()

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
    os.nice(19)
    setup(sys.argv[1])
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
