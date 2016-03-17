# sudo pip install dipy

from __future__ import print_function

#from dipy.align.aniso2iso import resample

import nipype.interfaces.ants as ants
#import nipype.interfaces.dipy as dipy
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

MULTITHREAD = "max" # 1,23,4....., "max"
# MULTITHREAD = "max"

DATA_PATH = ''
T1_PATTERN = []
DATA_OUT_PATH = ''
TEMP_FOLDER_PATH = 'temp/'
TEMPLATE_VOLUME = ""
TEMPLATE_MASK = ""

os.environ['FSLOUTPUTTYPE'] = 'NIFTI'
    
def prepareTemplate():
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
#    reslice.inputs.out_file = TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_temp.nii.gz'
#    reslice.run()

    # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:
    bet = fsl.BET(command="fsl5.0-bet")
    bet.inputs.in_file = data
    bet.inputs.frac = 0.45 # fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates
    bet.inputs.vertical_gradient = 0 # vertical gradient in fractional intensity threshold (-1->1); default=0; positive values give larger brain outline at bottom, smaller at top
    bet.inputs.reduce_bias = True #  This attempts to reduce image bias, and residual neck voxels. This can be useful when running SIENA or SIENAX, for example. Various stages involving FAST segmentation-based bias field removal and standard-space masking are combined to produce a result which can often give better results than just running bet2.
    bet.inputs.out_file = TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_bet.nii.gz'
    if os.path.exists(bet.inputs.out_file ):
        return bet.inputs.out_file
    bet.run()
    print(TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_bet.nii.gz')
    return bet.inputs.out_file

def registration(moving, fixed):
    """Image2Image registration """
    reg = ants.Registration()
    reg.inputs.collapse_output_transforms=True
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.num_threads=1
    reg.inputs.transforms = ['Rigid', 'Affine'] #, 'SyN']
    reg.inputs.winsorize_lower_quantile=0.005
    reg.inputs.winsorize_upper_quantile=0.995
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
    reg.inputs.sigma_units=['vox']*3
    reg.inputs.transform_parameters = [(0.1,),
                                       (0.1,),
                                       (0.2, 3.0, 0.0)]
    reg.inputs.use_histogram_matching = True
    reg.inputs.write_composite_transform = True
    #reg.inputs.collapse_output_transforms = False
    
    name=splitext(splitext(basename(moving))[0])[0]
    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + "output_"+name+'_'
    reg.inputs.output_warped_image = TEMP_FOLDER_PATH + name + '_reg.nii'
    #print(reg.cmdline)
    res= reg.inputs.output_transform_prefix + 'Composite.h5'
    if os.path.exists(res):
        return res
    out=reg.run()
    
    img=nib.load(reg.inputs.output_warped_image).get_data()
    img_template = nib.load(TEMPLATE_VOLUME).get_data()

    def show_slices(slices,layers):
        fig,axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
            axes[i].imshow(layers[i].T, cmap=cm.Reds, origin="lower", alpha=0.4)

    x=img.shape[0]/2
    y=img.shape[1]/2
    z=img.shape[2]/2
    
    slice_0=img[x, :, :]
    slice_1=img[:, y, :]
    slice_2=img[:, :, z]
    slices=[slice_0, slice_1, slice_2]
    
    slice_0=img_template[x, :, :]
    slice_1=img_template[:, y, :]
    slice_2=img_template[:, :, z]
    slices_template=[slice_0, slice_1, slice_2]

    show_slices(slices,slices_template)
    plt.suptitle(name)

    return res

def move_data(moving, transform):
    """ Move data with transform """
    at = ants.ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = moving
    at.inputs.reference_image = TEMPLATE_VOLUME
    at.inputs.output_image = DATA_OUT_PATH + splitext(basename(moving))[0] + '_reg.nii'
    at.inputs.interpolation = 'NearestNeighbor'
    at.inputs.default_value = 0
    at.inputs.transforms = [transform]
    at.inputs.invert_transform_flags = [False]
    #print(at.cmdline)
    at.run()

    return at.inputs.output_image


def post_calculation(images, label):
    avg = ants.AverageImages()
    avg.inputs.dimension = 3
    avg.inputs.output_average_image = DATA_OUT_PATH + 'avg_' + label + '.nii'
    avg.inputs.normalize = True
    avg.inputs.images = images
    print(avg.cmdline)
    avg.run()

def find_moving_images():
    res =[]
    for pattern in T1_PATTERN:
        res.extend(glob.glob(DATA_PATH +  '*' + pattern +'*'))
    return res

def find_seg_images(moving):
    pattern = ''
    for char in basename(moving)[1:]:
        if char == '-':
            break
        pattern += str(char)
    res = glob.glob(os.path.dirname(moving) + '/k' + pattern + '*.nii')
    if len(res) ==0: #LGG
        pattern = os.path.splitext(os.path.basename(moving))[0]
        res = glob.glob(os.path.dirname(moving) + '/'+pattern + '*.nii')
#    res.remove(moving)
    return res

def find_label(path):
     label = splitext(basename(path))[0]
     label = '_'.join(label.split("_")[1:])
     return label

def process_dataset(moving):
    print(moving)
    moving_preProcessed = pre_process(moving)
    transform = registration(moving_preProcessed, TEMP_FOLDER_PATH + "masked_template.nii")
    return (moving, transform)

def move_dataset(moving_dataset):
    if MULTITHREAD > 1:
        if MULTITHREAD == 'max':
            pool = Pool()
        else:
            pool = Pool(MULTITHREAD)
        res = pool.map_async(process_dataset, moving_dataset).get(9999999) # http://stackoverflow.com/a/1408476/636384
        pool.close()
        pool.join()
    else:
        res = map(process_dataset, moving_dataset)
    return res

def move_segmentations(transforms):
    res = dict()
    for moving,transform in transforms:
        for segmentation in find_seg_images(moving):
            print("         ", segmentation, transform)
            temp = move_data(segmentation,  transform)
            label = find_label(temp)
            if label in res:
                res[label].append(temp)
            else:
                res[label] = [temp]
    return res

def setup(dataset):
    global DATA_PATH, T1_PATTERN, DATA_OUT_PATH, TEMP_FOLDER_PATH, TEMPLATE_VOLUME, TEMPLATE_MASK
    hostname = os.uname()[1]
    if hostname == 'dahoiv-Alienware-15':
        if dataset=="HGG":
            DATA_PATH = '/home/dahoiv/disk/data/tumor_segmentation/'
            T1_PATTERN = ['T1_diag', 'T1_preop']
            DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/out/'
            TEMP_FOLDER_PATH = 'temp/'
        elif dataset=="LGG":
            DATA_PATH = '/home/dahoiv/disk/data/LGG_kart/PRE/'
            T1_PATTERN = ['_pre.nii']
            DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/out_LGG/'
            TEMP_FOLDER_PATH = 'temp_LGG/'
        else:
            print("Unkown dataset")
            raise Exception
        TEMPLATE_VOLUME = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
        TEMPLATE_MASK = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
        os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/' #path to ANTs bin folder
    else:
        print("Unkown host name " + hostname)
        print("Add your host name path to " +sys.argv[0] )
        raise Exception

if __name__ == "__main__":
    os.nice(19)
    setup(sys.argv[1])
    if not os.path.exists(TEMP_FOLDER_PATH):
        os.makedirs(TEMP_FOLDER_PATH)
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    prepareTemplate()

    moving_datasets = find_moving_images()
    transforms = move_dataset(moving_datasets)
    res = move_segmentations(transforms)

    for label in res:
        post_calculation(res[label], label)



        #break
