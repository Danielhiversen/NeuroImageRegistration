from __future__ import print_function

from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import AverageImages
from os.path import basename
from os.path import splitext
import os
import glob
import nipype.interfaces.fsl as fsl
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


DATA_PATH = '/home/dahoiv/disk/data/tumor_segmentation/'
T1_PATTERN = 'T1_diag.nii'
FIXED_VOLUME = "/home/dahoiv/disk/sintef/NeuroImageRegistration/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
os.environ["PATH"] += os.pathsep + '/home/dahoiv/disk/kode/ANTs/antsbin/bin/' #path to ANTs bin folder
DATA_OUT_PATH = '/home/dahoiv/disk/sintef/NeuroImageRegistration/out/'
TEMP_FOLDER_PATH = 'temp/'

def pre_process(data):
    """ Pre process the data"""
    bet = fsl.BET(command="fsl5.0-bet2")
    bet.inputs.in_file = data
    bet.inputs.out_file = TEMP_FOLDER_PATH + splitext(basename(data))[0] + '_pre.nii.gz'

    #print(bet.cmdline)
#
#    import nipype.interfaces.dipy as dipy
#    reslice = dipy.Resample()
#    reslice.inputs.in_file = 'diffusion.nii'
#    reslice.run() 
#

    bet.run()
    return bet.inputs.out_file

def registration(moving, fixed):
    """Image2Image registration """
    reg = Registration()
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving
    reg.inputs.output_transform_prefix = TEMP_FOLDER_PATH + "output_"
#    reg.inputs.initial_moving_transform = 'trans.mat'
#    reg.inputs.invert_initial_moving_transform = True
    reg.inputs.transforms = ['Affine']
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['Mattes']
    reg.inputs.metric_weight = [1] # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.05]
    reg.inputs.convergence_threshold = [1.e-8]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[1,0]]
    reg.inputs.sigma_units = ['vox'] 
    reg.inputs.shrink_factors = [[2,1]]
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True] # This is the default
    reg.inputs.output_warped_image = DATA_OUT_PATH + splitext(basename(moving))[0] + '_reg.nii'
    #print(reg.cmdline)
    out = reg.run()    
    
    return out.outputs.forward_transforms

def move_data(moving, transform):
    """ Move data with transform """
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = moving
    at.inputs.reference_image = moving
    at.inputs.output_image = DATA_OUT_PATH + splitext(basename(moving))[0] + '_reg.nii'
    at.inputs.interpolation = 'Linear'
    at.inputs.default_value = 0
    at.inputs.transforms = [transform]
    at.inputs.invert_transform_flags = [False]
    #print(at.cmdline)
    at.run()    
    
    return at.inputs.output_image


def post_process(images, label):
    avg = AverageImages()
    avg.inputs.dimension = 3
    avg.inputs.output_average_image = DATA_OUT_PATH + 'avg_' + label + '.nii'
    avg.inputs.normalize = True
    avg.inputs.images = images
    print(avg.cmdline)
    avg.run()

def find_moving_images(path, pattern):
    return glob.glob(path +  '*' + pattern)

def find_seg_images(moving):
    pattern = ''
    for char in basename(moving)[1:]:
        if char == '-':
            break
        pattern += str(char)
    res = glob.glob(os.path.dirname(moving) + '/k' + pattern + '*.nii')
    res.remove(moving)
    return res
    
def find_label(path):
     label = splitext(basename(path))[0]
     label = '_'.join(label.split("_")[1:])
     return label

def process_dataset(moving):
    print(moving)
    moving_preProcessed = pre_process(moving)
    transform = registration(moving_preProcessed, FIXED_VOLUME)
    return (moving, transform)        

def move_dataset(moving_dataset):
    res = map(process_dataset, moving_dataset)
    return res

def move_segmentations(transforms):
    res = dict()
    for moving,transform in transforms:    
        for segmentation in find_seg_images(moving):
            print("         ", segmentation)
            temp = move_data(segmentation,  transform[0])
            label = find_label(temp)
            if label in res:
                res[label].append(temp)
            else:
                res[label] = [temp]
    return res                
                
if __name__ == "__main__":
    if not os.path.exists(TEMP_FOLDER_PATH):
        os.makedirs(TEMP_FOLDER_PATH)
    moving_datasets = find_moving_images(DATA_PATH, T1_PATTERN)[0:2]
    transforms = move_dataset(moving_datasets)
    res = move_segmentations(transforms)

    for label in res:
        post_process(res[label], label)
        
        
            
        #break
