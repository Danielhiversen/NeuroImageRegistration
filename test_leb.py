import nibabel as nib
import util
from scipy.spatial import distance
import numpy as np

atlas_path = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resample.nii.gz"
label = 49

c, ci =  util.get_center_of_mass(atlas_path)
c_label, ci_label =  util.get_center_of_mass(atlas_path,49)

#coordinates1 = util.get_label_coordinates(atlas_path,22)
#coordinates2 = util.get_label_coordinates(atlas_path,23)

print('Done 1')

#dist = distance.cdist(coordinates1,coordinates2,'euclidean')

print('Done 2')

#print(np.amin(dist))

print('Done 3')

coords1 = [(5, 14, 6), (3, 9, 16), (18,2,15), (9,1,1)]
coords1 = [(5, 14, 6), (3, 9, 16), (18,2,15), (9,1,1)]
coords2 = [(3, 6, 8), (6,9,1)]
dist1 = distance.cdist(coords1, coords1, 'euclidean')

print(c_label)
print(distance.cdist(coords1, [c_label], 'euclidean'))
print(distance.cdist(coords1, [c_label], 'euclidean').min())





#disp(c)
#disp(ci)
#disp(c_label)
#disp(ci_label)
