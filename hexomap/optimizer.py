# this file contains optimizer like twiddle and line search

def twiddle_optimize(func, p, dp, threshold):
    """
    twiddle optimizer
    :param func: the loss function, larger is worse, reach minimum at optimal value
    :param p: list, initial value,
    :param dp: list, initial range
    :param threshold:, if sum(dp)<threshold, return p
    :return: pOptimal: found optimal value
    reference:
    https://martin-thoma.com/twiddle/
    """
    # Calculate the error
    best_err = func(p)

    while sum(dp) > threshold:
        for i in range(len(p)):
            p[i] += dp[i]
            err = func(p)

            if err < best_err:  # There was some improvement
                best_err = err
                dp[i] *= 1.1
            else:  # There was no improvement
                p[i] -= 2 * dp[i]  # Go into the other direction
                err = func(p)

                if err < best_err:  # There was an improvement
                    best_err = err
                    dp[i] *= 1.1
                else:  # There was no improvement
                    p[i] += dp[i]
                    # As there was no improvement, the step size in either
                    # direction, the step size might simply be too big.
                    dp[i] *= 0.9
    return p

