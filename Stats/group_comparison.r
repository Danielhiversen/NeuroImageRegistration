library(oro.nifti)
library(fslr)
library(foreach)
library(abind)
#library(doMC)
#registerDoMC(cores=30)
library(doParallel)

host <- Sys.info()["nodename"]
if (host == 'SINTEF-0ZQHTDG'){
    n_cores = 4
    path_name <- '/Users/leb/OneDrive - SINTEF/Prosjekter/Nevro/Brain atlas/Data/RES_survival_time_20191018_1156/'
}
else if (host == 'medtech-beast') {
    n_cores = 30
    path_name <- ''
}
else

registerDoParallel(cores=n_cores)

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
print(paste('Processing array',i, '/', img_dim[1]))
acomb <- function(...) abind(..., along=3)
t = system.time({
    p_values <-
        foreach(k = 1:img_dim[3], .combine='acomb', .multicombine=TRUE) %:% 
            foreach(j = 1:img_dim[2], .combine='cbind') %dopar% {
                if (j == 1) print(paste('Processing array',k, '/', img_dim[3]))
                temp = rep(0,img_dim[1])
                for(i in 1:img_dim[1]){ 
                    if(valid_voxels[i,j,k]){
                        t <- c(
                            low_survival_img[i,j,k],#4
                            medium_survival_img[i,j,k],#7
                            high_survival_img[i,j,k],#7
                            low_survival_n - low_survival_img[i,j,k],
                            medium_survival_n - medium_survival_img[i,j,k],
                            high_survival_n - high_survival_img[i,j,k]
                        )
                        cont_table <- matrix( t, nrow=2, byrow=TRUE )

                        ## Long version (~17% slower)
                        #rownames <- c('Tumor', 'No tumor')
                        #colnames <- c('Low', 'Medium', 'High')
                        #cont_table <- matrix( t, nrow=2, byrow=TRUE, dimnames = list(rownames,colnames) )

                        res <- fisher.test(cont_table)
                        temp[i] <- res$p.value
                    }
                }
                a<-temp

                # temp = c(0,img_dim[1])
                # for(i in 1:img_dim[1]){ 
                #     temp[i] = medium_survival_img[i,j,k]
                # }
                # a <- temp
            }
})
print(t[3])

p_values_img <- niftiarr(low_survival_img, p_values)
writeNIfTI(p_values_img, filename=paste0(path_name,'p_values_i1'))
