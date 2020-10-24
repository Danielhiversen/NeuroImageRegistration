library(oro.nifti)
library(fslr)
library(fdrtool)

results_folder_name <- 'RES_survival_stats_20200403_2134'
p_values_file_name <- 'p_values_corrected.nii.gz'
q_values_file_name <- 'q_values'

p_values_img <- readNIfTI( paste(results_folder_name, p_values_file_name, sep='/') )
img_dim <- p_values_img@dim
p_values <- p_values_img[p_values_img>0]

#fdr.estimate.eta0(p_values, method="smoother")
fdr <- fdrtool(p_values, statistic="pvalue")

q_values_array <- array(0, img_dim)
q_values_array[p_values_img>0] <- fdr$qval
q_values_img <- niftiarr(p_values_img, q_values_array)
writeNIfTI(q_values_img, filename=paste(results_folder_name, q_values_file_name, sep='/')) 
