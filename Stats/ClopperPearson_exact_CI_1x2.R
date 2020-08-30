ClopperPearson_exact_CI_1x2 = function(X, n, alpha=0.05, printresults=T) {

    # The Clopper-Pearson exact confidence interval for the binomial probability
    # Described in Chapter 2 "The 1x2 Table and the Binomial Distribution"
    # 
    # Input arguments
    # ---------------
    # X: the number of successes
    # n: the total number of observations
    # alpha: the nominal level, e.g. 0.05 for 95# CIs 
    # printresults: display results (F = no, T = yes)

    if (missing(n)) {
    #    X = 250; n = 533  # Example: The number of 1st order male births (Singh et al. 2010)
    #    X = 204; n = 412  # Example: The number of 2nd order male births (Singh et al. 2010)
    #    X = 103; n = 167  # Example: The number of 3rd order male births (Singh et al. 2010)
    #    X = 33; n = 45    # Example: The number of 4th order male births (Singh et al. 2010)
        X = 13; n = 16     # Example: Ligarden et al. (2010)
    }

    # Define global variables that are needed in the functions below
    # global Xglobal nglobal alphaglobal
    # Xglobal = X;
    # nglobal = n;
    # alphaglobal = alpha;

    # Estimate of the binomial probability (pihat)
    estimate = X/n

    # Use Matlabs fzero function to solve the equations for the confidence limits
    tol = 0.00000001
    # options = optimset('Display', 'off', 'TolX', tol);

    # Find the lower CI limit
    if (estimate == 0) {
        L = 0
    } else {
        L = uniroot(calculate_L_CP, interval=c(0,1), X=X, n=n, alpha=alpha, tol=tol)$root
    }

    # Find the upper CI limit
    if (estimate == 1) {
        U = 1
    } else {
        U = uniroot(calculate_U_CP, interval=c(0,1), X=X, n=n, alpha=alpha, tol=tol)$root
    }

    if (printresults) {
        print(sprintf('The Clopper Pearson exact CI: estimate = %6.4f (%g%% CI %6.4f to %6.4f)', 
            estimate, 100*(1 - alpha), L, U), quote=F)
    }
    
    res = c(L, U, estimate)
    names(res) = c("lower", "upper", "estimate")
    invisible(res)
    
}

# ==========================
calculate_L_CP = function(L0, X, n, alpha) {
    # global Xglobal nglobal alphaglobal
    T0 = sum(dbinom(X:n, n, L0))
    L = T0 - alpha/2
    return(L)
}

# ==========================
calculate_U_CP = function(U0, X, n, alpha) {
    # global Xglobal nglobal alphaglobal
    T0 = sum(dbinom(0:X, n, U0))
    U = T0 - alpha/2
    return(U)
}

