.print = function(s, ...) {
		print(sprintf(gsub('\n','',s), ...), quote=F)
	}
	
# function [P, midP] = Exact_cond_midP_unspecific_ordering_rx2(n, direction, 
# statistic, printresults)

Exact_cond_midP_unspecific_ordering_rx2 = function(n, direction, statistic="Pearson", printresults=T) {
	# The exact conditional and mid-P tests for unspecific ordering
	# Described in Chapter 5 "The Ordered rx2 Table"
	#
	# May also be used for 2xc tables, after flipping rows and columns (i.e. if
	# n is a 2xc table, call this function with n' (the transpose of n) as the first argument)
	#
	# Input arguments
	# ---------------
	# n: the observed counts (an rx2 matrix)
	# direction: the direction of the success probabilities ("increasing" or 
	#            "decreasing")
	# statistic: the Pearson test statistic ("Pearson") or the likelihood ratio
	#            test statistic ("LR"). Can also be used for cumulative ORs in
	#            2xc tables with "PearsonCumOR" or "LRCumOR".
	# printresults: display results (0 = no, 1 = yes)
	
	if (missing(direction)) {	
	    # Chapter 6: Postoperative nausea (Lydersen et al., 2012a)
	    n = t(rbind(c(14, 10, 3, 2), c(11, 7, 8, 4)))
	    direction = "decreasing"
	    statistic = "PearsonCumOR"
	}
	
	r = nrow(n)
	nip = apply(n, 1, sum)
	npj = apply(n, 2, sum)
	N = sum(n)
	np1 = sum(n[,1])
	
	# Calculate all nchoosek beforehand
	nip_choose_xi1 = matrix(0, r, max(nip)+1)
	for (i in 1:r) {
	    for (xi1 in 0:nip[i]) {
	        nip_choose_xi1[i, xi1+1] = choose(nip[i], xi1)
	    }
	}
	N_choose_np1 = choose(N, np1)
	
	# The observed value of the test statistic
	Tobs = test_statistic(n, r, nip, npj, N, direction, statistic)
	
	# Calculate the two-sided exact P-value and the mid-P value
	# Need separate functions for different values of r (the number of rows)
	if (r == 3) {
	    tmp = calc_Pvalue_3x2(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic)
	    P = tmp$P
	    midP = tmp$midP
	} else if (r == 4) {
	    tmp = calc_Pvalue_4x2(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic)
	    P = tmp$P
	    midP = tmp$midP
	} else if (r == 5) {
	    tmp = calc_Pvalue_5x2(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic)
	    P = tmp$P
	    midP = tmp$midP
	}
	
	if (printresults) {
	    .print('Exact conditional test: %8.5f\n', P)
	    .print('Mid-P test:             %8.5f\n', midP)
	}
	
	invisible(data.frame(P=P, midP=midP))
}
	
# Brute force calculations of the two-sided exact P-value and the mid-P value
# This function assumes r=3 rows

calc_Pvalue_3x2 = function(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic) {
	P = 0
	point_prob = 0
	for (x1 in 0:min(c(nip[1], np1))) {
	    for (x2 in 0:min(c(nip[2], np1-x1))) {
			x3 = np1 - x1 - x2
			if (x3 > nip[3]) {
				next
			}
			x = rbind(c(x1, nip[1]-x1), c(x2, nip[2]-x2), c(x3, nip[3]-x3))
			T0 = test_statistic(x, 3, nip, npj, N, direction, statistic)
			f = calc_prob(x[,1], 3, N_choose_np1, nip_choose_xi1)
			if (T0 == Tobs) {
				point_prob = point_prob + f
			} else if (T0 > Tobs) {
				P = P + f
			}  
	    }
	}
	midP = P + 0.5 * point_prob
	P = P + point_prob
	
	return(data.frame(P=P, midP=midP))
}

# Brute force calculations of the two-sided exact P-value and the mid-P value
# This function assumes r=4 rows

calc_Pvalue_4x2 = function(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic) {
	P = 0
	point_prob = 0
	for (x1 in 0:min(c(nip[1], np1))) {
	    for (x2 in 0:min(c(nip[2], np1-x1))) {
	        for (x3 in 0:min(c(nip[3], np1-x1-x2))) {
	            x4 = np1 - x1 - x2 - x3
	            if (x4 > nip[4]) {
	            	next
	            }
	            x = rbind(c(x1, nip[1]-x1), c(x2, nip[2]-x2), c(x3, nip[3]-x3), c(x4, nip[4]-x4))
	            T0 = test_statistic(x, 4, nip, npj, N, direction, statistic)
	            f = calc_prob(x[,1], 4, N_choose_np1, nip_choose_xi1)
	            if (T0 == Tobs) {
	                point_prob = point_prob + f
	            } else if (T0 > Tobs) {
	                P = P + f
	          	}  
	        }
	    }
	}
	midP = P + 0.5 * point_prob
	P = P + point_prob
	
	return(data.frame(P=P, midP=midP))
}
	
# Brute force calculations of the two-sided exact P-value and the mid-P value
# This function assumes r=5 rows

calc_Pvalue_5x2 = function(Tobs, nip, np1, npj, N, N_choose_np1, nip_choose_xi1, direction, statistic) {
	P = 0
	point_prob = 0
	for (x1 in 0:min(c(nip[1], np1))) {
	    for (x2 in 0:min(c(nip[2], np1-x1))) {
	        for (x3 in 0:min(c(nip[3], np1-x1-x2))) {
	            for (x4 in 0:min(c(nip[4], np1-x1-x2-x3))) {
	                x5 = np1 - x1 - x2 - x3 - x4
	                if (x5 > nip[5]) {
	                	next
	                }
	                x = rbind(c(x1, nip[1]-x1), c(x2, nip[2]-x2), c(x3, nip[3]-x3), c(x4, nip[4]-x4), c(x5, nip[5]-x5))
	                T0 = test_statistic(x, 5, nip, npj, N, direction, statistic)
	                f = calc_prob(x[,1], 5, N_choose_np1, nip_choose_xi1)
	                if (T0 == Tobs) {
	                    point_prob = point_prob + f
	                } else if (T0 > Tobs) {
	                    P = P + f
	                }
	            }
	        }
	    }
	}
	midP = P + 0.5*point_prob;
	P = P + point_prob;
	
	return(data.frame(P=P, midP=midP))
}
	
	
# Calculate the test statistics
test_statistic = function(n, r, nip, npj, N, direction, statistic) {
	# These are used for cumulative odds ratios in 2xc tables
	if (identical(statistic,'PearsonCumOR') || identical(statistic,'LRCumOR')) {
	    n = t(n)
	    n[c(1,2),] = n[c(2,1),]
	    T0 = test_statistic_cum_OR(n, r, nip, statistic)
	    return(T0)
	}
	
	# Common calculations for the Pearson and LR statistics
	nhat = n[,1]/apply(n, 1, sum)
	nhatstar = nhat
	for (i in 1:(r-1)) {
	    if ((identical(direction,'increasing') && nhatstar[i]>nhatstar[i+1]) || 
	       (identical(direction,'decreasing') && nhatstar[i] < nhatstar[i+1])) {
	         pooled_proportion = (n[i,1]+n[i+1,1])/(n[i,1]+n[i,2]+n[i+1,1]+n[i+1,2])
	         nhatstar[i] = pooled_proportion
	         nhatstar[i+1] = pooled_proportion
	    }
	}
	nstar = matrix(0, r, 2)
	nstar[,1] = apply(n, 1, sum) * nhatstar
	nstar[,2] = apply(n, 1, sum) * (1 - nhatstar)
	
	m = matrix(0, r, 2)
	T0 = 0
	if (identical(statistic, 'Pearson')) {
	    for (i in 1:r) {
	        for (j in 1:2) {
	            m[i,j] = nip[i]*npj[j]/N
	            if (m[i,j] > 0) {
	                T0 = T0 + ((nstar[i,j] - m[i,j])^2)/m[i,j]
	            }
	        }
	    }
	} else if (identical(statistic, 'LR')) {
	    for (i in 1:r) {
	        for (j in 1:2) {
	            m[i,j] = nip[i]*npj[j]/N
	            if (m[i,j] > 0 && nstar[i,j] != 0) {
	                T0 = T0 + nstar[i,j]*log(nstar[i,j]/m[i,j]);
	            }
	        }
	    }
	    T0 = 2 * T0
	}
	
	return(T0)
}
	
# Slightly different calculations are needed for cumulative odds ratios in 2xc tables
	
test_statistic_cum_OR = function(n, c, npj, statistic) {
	r = rep(0, c)
	for (j in 1:c) {
	    r[j] = sum(n[1,1:j])/sum(n[2,1:j])
	}
	J = list(); index1 = 1
	while (index1 < c+1) {
	    # [~, v] = min(r(index1:end));
	    v = which(r[index1:length(r)] == min(r[index1:length(r)]))[1]
	    J[[length(J)+1]] = n[,index1:(index1+v-1),drop=F]
	    index1 = index1 + v
	}	
	
	m = array(0, dim=c(length(J), 2, c))
	T0 = 0
	if (identical(statistic, 'PearsonCumOR')) {
	    for (h in 1:length(J)) {
	        nJ = J[[h]]
	        cols = ncol(nJ)
	        for (i in 1:2) {
	            for (j in 1:cols) {
	                # m[h,i,j] = npj[j] * sum(nJ[i,], 2)/sum(nJ)
	                m[h,i,j] = npj[j] * sum(nJ[i,])/sum(nJ)
	                if (m[h,i,j] > 0) {
	                    T0 = T0 + ((n[i,j] - m[h,i,j])^2)/m[h,i,j];
	                }
	            }
	        }
	    }
	} else if (identical(statistic, 'LRCumOR')) {
	    for (h in 1:length(J)) {
	        nJ = J[[h]]
	        cols = ncol(nJ)
	        for (i in 1:2) {
	            for (j in 1:cols) {
	                # m[h,i,j] = npj[j] * sum(nJ[i,], 2)/sum(nJ)
	                m[h,i,j] = npj[j] * sum(nJ[i,])/sum(nJ)
	                if (m[h,i,j] > 0 && n[i,j] > 0) {
	                    T0 = T0 + n[i,j]*log(n[i,j]/m[h,i,j])
	                }
	            }
	        }
	    }
	    T0 = 2 * T0
	}
	
	return(T0)
}
	
# Calculate the probability of table x
# (multiple hypergeometric distribution)

calc_prob = function(x, r, N_choose_np1, nip_choose_xi1) {
	f = 1
	for (i in 1:r) {
	    f = f * nip_choose_xi1[i, x[i]+1]
	}
	f = f/N_choose_np1
	return(f)
}
	
	