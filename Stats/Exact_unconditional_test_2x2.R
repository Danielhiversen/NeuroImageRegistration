##### IN LINE 71: ADDED !is.na(..) TO AVOID CRASH ####

source('ClopperPearson_exact_CI_1x2.R')
source('Fisher_exact_test_2x2.R')

Exact_unconditional_test_2x2 = function(n, statistic='Pearson', gamma=0.0001, printresults=T) {

    # function P = Exact_unconditional_test_2x2(n, statistic, gamma, printresults)
    # Exact unconditional test for association in 2x2 tables
    # Described in Chapter 4 "The 2x2 Table"
    #
    # Dependencies: "ClopperPearson_exact_CI_1x2_beta_version.m", "Fisher_exact_test_2x2.m"
    #
    # Input arguments
    # ---------------
    # n: the observed counts (a 2x2 matrix)
    # statistic: 'Pearson' (Suissa-Shuster test default), 'LR' (likelihood ratio), 
    #    'unpooled' (unpooled Z), or 'Fisher' (Fisher-Boschloo test)
    # gamma: parameter for the Berger and Boos procedure (default=0.0001 gamma=0: no adj)
    # printresults: display results (F = no, T = yes)
    #
    # Somewhat crude code with maximization over a simple partition of the
    # nuisance parameter space into 'num_pi_values' equally spaced values
    # (default: 1000). The number may be changed below.
    # This method could be improved with a better algorithm for the 
    # maximization however, it works well for most purposes. Try 'showplot=TRUE' 
    # to get an indication of the precision. A refinement of the maximization 
    # can be done with a manual restriction of the parameter space.

    if (missing(n)) {
        n = rbind(c(3,1), c(1,3))       # Example: A lady tasting a cup of tea
        # n = rbind(c(7,27), c(1,33))   # Example: Perondi et al. (2004)
        # n = rbind(c(9,4), c(4,10))    # Example: Lampasona et al. (2013)
        # n = rbind(c(0,16), c(15,57))  # Example: Ritland et al. (2007)
    }

    # Partition the parameter space into 'num_pi_values' equally spaced values
    num_pi_values = 1000

    # Display a plot of the P-value as a function of the common success probability
    showplot = FALSE

    n1p = n[1,1] + n[1,2]
    n2p = n[2,1] + n[2,2]
    np1 = n[1,1] + n[2,1]
    N = sum(n)

    # Calculate and store the binomial coefficients needed
    # warning('off', 'all')
    binomcoeffs1 = rep(0, n1p+1)
    for (x11 in 0:n1p) {
        binomcoeffs1[x11+1] = choose(n1p, x11)
    }
    if (n1p == n2p) {
        binomcoeffs2 = binomcoeffs1
    } else {
        binomcoeffs2 = rep(0, n2p+1)
        for (x21 in 0:n2p) {
            binomcoeffs2[x21+1] = choose(n2p, x21)
        }
    }
    binomcoeffs = outer(binomcoeffs1, binomcoeffs2)
    # warning('on', 'all')

    # Find the tables that agree equally or less with H0 as the observed
    tables = matrix(0, n1p+1, n2p+1)
    Tobs = test_statistic_exact_unconditional_test_2x2(n[1,1], n[1,2], n[2,1], n[2,2], statistic)
    for (x11 in 0:n1p) {
        for (x21 in 0:n2p) {
            T0 = test_statistic_exact_unconditional_test_2x2(x11, n1p-x11, x21, n2p-x21, statistic)
            if (!is.na(T0) && T0 >= Tobs) {  ##### ADDED !is.na(..) TO AVOID CRASH ####
                tables[x11+1, x21+1] = 1
            }
        }
    }

    # A simple partition the nuisance parameter space
    if (gamma == 0) {
        pivalues = seq(0, 1, length=num_pi_values)
    } else {
        # Berger and Boos procedure
        # Use the Clopper-Pearson exact interval
        # [L, U, ~] = ClopperPearson_exact_CI_1x2_beta_version(np1, N, gamma, 0)
        res.cpe = ClopperPearson_exact_CI_1x2(np1, N, gamma, printresults=F)
        pivalues = seq(getElement(res.cpe,'lower'), getElement(res.cpe,'upper'), length=num_pi_values)
    }

    # Calculate the P-value corresponding to each value of the nuisance parameter
    Pvalues = rep(0, length(pivalues))
    for (i in 1:length(pivalues)) {
        Pvalues[i] = calculate_Pvalue(pivalues[i], tables, binomcoeffs, n1p, n2p)
    }

    # Let the exact unconditional P-value equal the maximum of the P-values
    # [P, index] = max(Pvalues)
    P = max(Pvalues)
    index = which(P==Pvalues)[1]

    # Add gamma (the parameter for the Berger and Boos procedure) to make sure
    # the actual significance level is bounded by the nominal level
    P = min(c(P + gamma, 1))

    # Handle cases where the P-value is not computable
    if (sum(tables) == 0) {
        P = 1.0
    }

    # Display a plot of the P-value as a function of the common success probability
    if (showplot) {
        common_pi_at_max_value = pivalues[index]
        quartz() # For Max OS X. Change to windows() to run under Windows.
        plot(pivalues, Pvalues, lty=1, col="black")
        lines(c(common_pi_at_max_value, common_pi_at_max_value), c(0, P), lty=2, col="red")
    }

    if (printresults) {
        if (statistic == 'Pearson') {
            print(sprintf('The Suissa-Shuster exact unconditional test: P = %7.5f', P), quote=F)
        } else if (statistic == 'LR') {
            print(sprintf('Exact unconditional test with the LR statistic: P = %7.5f', P), quote=F)
        } else if (statistic == 'unpooled') {
            print(sprintf('Exact unconditional test with the unpooled Z statistic: P = %7.5f', P), quote=F)
        } else if (statistic == 'Fisher') {
            print(sprintf('Fisher-Boschloo exact unconditional test: P = %7.5f', P), quote=F)
        }
    }
    
    invisible(P)

}


# ===================================================================
calculate_Pvalue = function(pi0, tables, binomcoeffs, n1p, n2p) {
    Pvalue = 0
    for (x11 in 0:n1p) {
        for (x21 in 0:n2p) {
            if (tables[x11+1, x21+1] == 1) {
                Pvalue = Pvalue + binomcoeffs[x11+1,x21+1] * (pi0^(x11+x21)) * ((1-pi0)^(n1p-x11+n2p-x21))
            }
        }
    }
    return(Pvalue)
}

# ========================================================
test_statistic_exact_unconditional_test_2x2 = function(x11, x12, x21, x22, statistic) {
    N = x11 + x12 + x21 + x22

    if (statistic == 'Pearson') {
        # The Pearson chi-squared statistic
        T0 = (N*(x11*x22-x12*x21)^2) / ((x11+x12)*(x21+x22)*(x11+x21)*(x12+x22))
    } else if (statistic == 'LR') {
        # The likelihood ratio statistic
        T0 = 0
        if (x11 > 0) {T0 = T0 + x11*log(x11/((x11 + x12)*(x11 + x21)/N))}
        if (x12 > 0) {T0 = T0 + x12*log(x12/((x11 + x12)*(x12 + x22)/N))}
        if (x21 > 0) {T0 = T0 + x21*log(x21/((x21 + x22)*(x11 + x21)/N))}
        if (x22 > 0) {T0 = T0 + x22*log(x22/((x21 + x22)*(x12 + x22)/N))}
        T0 = 2*T0
    } else if (statistic == 'unpooled') {
        # The unpooled Z statistic
        n1p = x11 + x12
        n2p = x21 + x22
        tmp1 = x11/n1p
        tmp2 = x21/n2p
        Z = (tmp1-tmp2) / sqrt((tmp1*(1 - tmp1)/n1p) + (tmp2*(1 - tmp2)/n2p))
        if (is.na(Z)) {Z = 0}
        T0 = Z^2
    } else if (statistic == 'Fisher') {
        # Fisher's exact test as test statistic
        x = matrix(c(x11, x12, x21, x22), nrow=2, byrow=T)
        T0 = -Fisher_exact_test_2x2(x, 'hypergeometric', printresults=FALSE)
    }
    
    return(T0)
    
}
