library(oro.nifti)
library(fslr)
library(foreach)
#library(doMC)
#registerDoMC(cores=30)
library(doParallel)
registerDoParallel(cores=30)

path_name <- ''
low_survival_file_name <- 'total_tumor_0-182.nii'
medium_survival_file_name <- 'total_tumor_183-730.nii'
high_survival_file_name <- 'total_tumor_183-730.nii'

low_survival_img <- readNIfTI(paste0(path_name,low_survival_file_name))
medium_survival_img <- readNIfTI(paste0(path_name,medium_survival_file_name))
high_survival_img <- readNIfTI(paste0(path_name,high_survival_file_name))

# Tall hentet fra artikkel. Dobbeltsjekk!
low_survival_n <- 52
medium_survival_n <- 123
high_survival_n <- 41

# Check that images have the same dimension
if (!( identical(low_survival_img@dim_,medium_survival_img@dim_) && identical(low_survival_img@dim_,medium_survival_img@dim_) )){
    stop('Images have different dimensions!')
}
img_dim <- low_survival_img@dim_[2:4]

valid_voxels <- (low_survival_img+medium_survival_img+high_survival_img)>9

#p_values_img <- niftiarr(low_survival_img, array(0,dim=img_dim))
p_values <- array(0,dim=img_dim)

# i <- 137
# j <- 84
# k <- 93
for(i in 1:img_dim[1]){
    print(paste('Processing array',i, '/', img_dim[1]))
    t = system.time({     
        m <-
            foreach(j = 1:img_dim[2], .combine='rbind') %:% 
                foreach(k = 1:img_dim[3], .combine='c') %dopar% {      
                    p_values[i,j,k] = medium_survival_img[i,j,k]

            }
    })
    print(t[3])
}
# for(i in 1:img_dim[1]){
#     print(paste('Processing array',i, '/', img_dim[1]))
#     t = system.time({     
#         m <-
#             foreach(j = 1:img_dim[2], .combine='rbind') %:% 
#                 foreach(k = 1:img_dim[3], .combine='c') %dopar% {      
#                     # medium_survival_img[i,j,k]
#                     #print(paste('Processing voxel', i, '/', img_dim[1], ', ', j, '/', img_dim[2], ', ', k, '/', img_dim[3]))
#                     if(valid_voxels[i,j,k]){
#                         t <- c(
#                             low_survival_img[i,j,k],#4
#                             medium_survival_img[i,j,k],#7
#                             high_survival_img[i,j,k],#7
#                             low_survival_n - low_survival_img[i,j,k],
#                             medium_survival_n - medium_survival_img[i,j,k],
#                             high_survival_n - high_survival_img[i,j,k]
#                         )
#                         rownames <- c('Tumor', 'No tumor')
#                         colnames <- c('Low', 'Medium', 'High')
#                         cont_table <- matrix( t, nrow=2, byrow=TRUE, dimnames = list(rownames,colnames) )

#                         res <- fisher.test(cont_table)
#                         p_value <- res$p.value
#                     }
#                     else{
#                         p_value <- 0
#                     }
#             }
#         p_values[i,,] = m
#     })
#     print(t[3])
# }
p_values_img <- niftiarr(low_survival_img, p_values)
writeNIfTI(p_values_img, filename=paste0(path_name,'p_values_i1'))
