library(jsonlite)
library(oro.nifti)
library(fslr)
library(foreach)
library(doParallel)
library(logging)

source('utils.r')

# File and folder names
results_folder_name <- format(Sys.time(),format='RES_survival_stats_low-medium_%Y%m%d_%H%M')
results_file_name <- list('p_values_original', 'p_values_corrected', 'q_values')
log_file_name <- 'group_comparison.log'
dir.create(results_folder_name)

# Setting up logging
logReset()
basicConfig(level='DEBUG')
addHandler(writeToFile, file=paste(results_folder_name, log_file_name, sep='/'), level='INFO')
removeHandler('writeToConsol')

# Host-specific parameters
host <- Sys.info()["nodename"]
if (host == 'SINTEF-0ZQHTDG'){
    library(fdrtool) # Bit available in R 3.4
    n_cores <- 4
} else if (host == 'medtech-beast') {
    n_cores <- 30
} else {
    logwarn('The computer named %s is not known. Number of cores is set to 1.', host)
    n_cores <- 1
}
registerDoParallel(cores=n_cores)

loginfo('Reading data')
pids_per_voxel <- fromJSON('pids_per_voxel.json')
#load('test_data.RData')
survival_group_per_patient <- unlist( fromJSON('survival_group_per_patient.json'), use.names=FALSE)

template_img_file <- 'total_tumor.nii.gz'
template_img <- readNIfTI(template_img_file)
img_dim <- template_img@dim_[2:4]

n_total <- count_patients_per_group(survival_group_per_patient)
n_permutations <- 2000
min_marginal <- 0

#results_folder <- '/Users/leb/OneDrive - SINTEF/Prosjekter/Nevro/Brain atlas/Data'
file <- c(
    'RES_survival_stats_low-high_20200913_2334/p_values_corrected_low-high.nii.gz',
    'RES_survival_stats_low-medium_20200913_0917/p_values_corrected_low-medium.nii.gz',
    'RES_survival_stats_medium-high_20200914_1213/p_values_corrected_medium-high.nii.gz'
)
precalculated_p_values <- vector()
for (f in file){
    #i <- readNIfTI(paste(results_folder, f, sep='/'))
    i <- readNIfTI(f)
    precalculated_p_values <- append(precalculated_p_values, as.vector(i[i>0]))
}

loginfo('Creating permutations')
loginfo('Number of permutations: %i', n_permutations)
set.seed(7)
permuted_indices <- rperm(n_permutations, length(survival_group_per_patient))

loginfo('Performing permutation tests')
batches_per_core <- 4
batch_size <- ((length(pids_per_voxel)/(batches_per_core*n_cores))%/%1000+1)*1000 # Rounded up to nearest 1000, leaving the last batch smaller than the rest. 
batch_lims <- seq(0,length(pids_per_voxel)-1, by=batch_size)
t1 <- system.time({
    results_array <- 
        foreach( lim = batch_lims, .combine = '+') %dopar% {
            lim1 <- lim+1
            lim2 <- min( lim+batch_size, length(pids_per_voxel) )
            batch <- c(lim1:lim2)
            t2 <- system.time({
                temp_array <- array(-1  , dim=c(3,img_dim))
                for (i in batch) {
                    pids <- pids_per_voxel[[i]]+1 # Add 1 to convert from pythonic, zero-based indexing
                    if (length(pids)>=min_marginal) {
                        res_original <- stat_test(survival_group_per_patient[pids], n_total)
                        p_value_original <- res_original$p
                        p_values <- rep(0, n_permutations)
                        for (j in 1:n_permutations) {
                            survival_groups_permuted <- survival_group_per_patient[permuted_indices[,j]]
                            res <- stat_test(survival_groups_permuted[pids], n_total)
                            p_values[j] <- res$p
                        }
                        p_value_corrected <- sum(p_values<p_value_original)/n_permutations
                        
                        #precalculated_p_values <- vector with all non-zero p_values from all three results files


                        n_tests <- length(precalculated_p_values)
                        n_false_discoveries <- n_tests*sum(p_values<p_value_corrected)/n_permutations
                        n_sigificant_voxels <- sum(precalculated_p_values<p_value_corrected)
                        q_value <- n_false_discoveries/n_sigificant_voxels

                        if( res_original$direction == 'increasing' ){
                            dir_sign <- 1
                        } else {
                            dir_sign <- -1
                        }

                        index_str <- names(pids_per_voxel[i])
                        index_str_list <- strsplit(index_str,'_')
                        index <- strtoi(unlist(index_str_list))+1 # Add 1 to convert from pythonic, zero-based indexing
                        temp_array[1, img_dim[1]+1-index[1], index[2], index[3]] <- p_value_original*dir_sign #p_values_corrected[[index_str]]                        
                        temp_array[2, img_dim[1]+1-index[1], index[2], index[3]] <- p_value_corrected*dir_sign #p_values_corrected[[index_str]]
                        temp_array[3, img_dim[1]+1-index[1], index[2], index[3]] <- q_value*dir_sign #p_values_corrected[[index_str]]
                    }                
                }
            })
            #cat(paste('Finished processing voxels', lim1, 'to', lim2, ' out of ', length(pids_per_voxel), ' in ', round(t2[3]), ' seconds.\n'), file='log.txt', append=TRUE)
            loginfo('Finished processing voxels %i to %i out of %i in %i seconds', lim1, lim2, length(pids_per_voxel), round(t2[3]))
            temp_array
        } 
})
loginfo('Total processing time: %i seconds.', round(t1[3]))

loginfo('Writing results to file')
for (i in 1:3){
    results_img <- niftiarr(template_img, results_array[i,,,])
    writeNIfTI(results_img, filename=paste(results_folder_name, results_file_name[[i]], sep='/'))
}

loginfo('Finished.')