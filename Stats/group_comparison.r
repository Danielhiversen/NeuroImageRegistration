library(oro.nifti)
library(fslr)

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

p_values_img <- niftiarr(low_survival_img, array(0,dim=img_dim))


# i <- 137
# j <- 84
# k <- 93

t = system.time({
    for (i in 1:img_dim[1]){
        j <- 1
        k <- 1
    #    for (j in 1:img_dim[2]){
    #        for (k in 1:img_dim[3]){
                print(paste('Processing voxel', c(i,j,k), 'of', img_dim))
                t <- c(
                    low_survival_img[i,j,k],#4
                    medium_survival_img[i,j,k],#7
                    high_survival_img[i,j,k],#7
                    low_survival_n - low_survival_img[i,j,k],
                    medium_survival_n - medium_survival_img[i,j,k],
                    high_survival_n - high_survival_img[i,j,k]
                )
                rownames <- c('Tumor', 'No tumor')
                colnames <- c('Low', 'Medium', 'High')
                cont_table <- matrix( t, nrow=2, byrow=TRUE, dimnames = list(rownames,colnames) )

                res <- fisher.test(cont_table)
                p_values_img[i,j,k] <- res$p.value
    #        }
    #    }
    }
})
print(t)
writeNIfTI(p_values_img, filename=paste0(path_name,'p_values_i1'))
