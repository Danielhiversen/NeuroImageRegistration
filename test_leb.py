#import nibabel as nib
import util

atlas_path = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resample.nii.gz"
label = 49

c, ci =  util.get_center_of_mass(atlas_path)
c_label, ci_label =  util.get_center_of_mass(atlas_path,49)

disp(c)
disp(ci)
disp(c_label)
disp(ci_label)
