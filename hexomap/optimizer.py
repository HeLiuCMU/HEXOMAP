#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Customized optimization routines for NF-HEDM reconstruction
    - twiddle search
    - line search
"""


from typing import Callable


def twiddle_optimize(func: Callable[..., float], 
                     p: list, 
                     dp: list, 
                     threshold: float,
        ) -> list:
    """
    Description
    -----------
        twiddle search optimizer

    Parameters
    ----------
    func: Callable
        the loss function, larger is worse, reach minimum at optimal value
    p: list
        initial parameter vector
    dp: list
        initial range for each parameter
    threshold: float 
        if sum(dp)<threshold, return p

    Returns
    -------
    list: 
        optimal value reported by twiddle search

    Reference
    ---------
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

