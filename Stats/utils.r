source('Exact_unconditional_test_2x2.R')
source('Exact_cond_midP_unspecific_ordering_rx2.R')

#' Perform statistical test of association.
#' 
#' @param groups_per_patient Vector with group number for each group.
#' @param total_per_group
#' @return p-value 
#' @examples
stat_test <- function(groups_per_patient, total_per_group) {
    n_total <- total_per_group
    n_tumor <- count_patients_per_group(groups_per_patient)
    t <- c(
        n_tumor$low,
        n_tumor$medium,
        #n_tumor$high,
        n_total$low - n_tumor$low,
        n_total$medium - n_tumor$medium#,
        #n_total$high - n_tumor$high
    )
    # t <- c(
    #     n_tumor$low,
    #     n_tumor$medium,
    #     n_tumor$low + n_tumor$medium,
    #     n_tumor$high,
    #     n_total$low - n_tumor$low,
    #     n_total$medium - n_tumor$medium,
    #     n_total$low - n_tumor$low + n_total$medium - n_tumor$medium,
    #     n_total$high - n_tumor$high
    #  )
    cont_table <- matrix( t, nrow=2, byrow=TRUE )
    
    ## Long version (~17% slower)
    #rownames <- c('Tumor', 'No tumor')
    #colnames <- c('Low', 'Medium', 'High')
    #cont_table <- matrix( t, nrow=2, byrow=TRUE, dimnames = list(rownames,colnames) )

    test <- 'Pearson' # Fisher or Pearson
    if (test=='Fisher'){
        res <- fisher.test(cont_table)
        p_value <- res$p.value
    } else {
        if (dim(cont_table)[2]==2) {
            res <- Exact_unconditional_test_2x2(cont_table, printresults=FALSE)
            result <- list('p'=res, 'direction'='')
        } else if (dim(cont_table)[2]>=3 & dim(cont_table)[2]<=5) {
            p1 <- n_tumor$low/n_total$low
            p2 <- n_tumor$medium/n_total$medium
            p3 <- n_tumor$high/n_total$high
            if ((p1<=p2 & p1>=p3) | (p1>=p2 & p1>=p3) ){
                direction <- 'decreasing'
            } else {
                direction <- 'increasing'
            }
            res <- Exact_cond_midP_unspecific_ordering_rx2(t(cont_table), direction,'Pearson', FALSE) #test=Pearson, LR, PearsonCumOR or LRCumOR

            if ((n_tumor$low+n_tumor$medium+n_tumor$high)==0){
                print('Voxel with no tumors')
                print(cont_table)
                print(res$P)
            }

            result <- list('p'=res$P, 'direction'=direction)
            #result <- list('p'=res$midP, 'direction'=direction)
        }
    }
}

count_patients_per_group <- function( group_per_patient ) {
    n <- list('low'=0,'medium'=0,'high'=0)
    for (i in 1:length(n)) n[i] <- sum(group_per_patient==i)
    n
}

# rperm creates m unique permutations of 1:size
# Returns a `size` by `m` matrix; each column is a permutation of 1:size.
#
# https://stats.stackexchange.com/questions/24300/how-to-resample-in-r-without-repeating-permutations
#
rperm <- function(m, size=2) { # Obtain m unique permutations of 1:size

    # Function to obtain a new permutation.
    newperm <- function() {
        count <- 0                # Protects against infinite loops
        repeat {
            # Generate a permutation and check against previous ones.
            p <- sample(1:size)
            hash.p <- paste(p, collapse="")
            if (is.null(cache[[hash.p]])) break

            # Prepare to try again.
            count <- count+1
            if (count > 1000) {   # 1000 is arbitrary; adjust to taste
                p <- NA           # NA indicates a new permutation wasn't found
                hash.p <- ""
                break
            }
        }
        cache[[hash.p]] <<- TRUE  # Update the list of permutations found
        p                         # Return this (new) permutation
    }

    # Obtain m unique permutations.
    cache <- list()
    replicate(m, newperm())  
} 