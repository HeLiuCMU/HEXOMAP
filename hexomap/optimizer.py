#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Customized optimization routines for NF-HEDM reconstruction
    - twiddle search
    - line search
"""


from typing import Callable


def twiddle_optimize(func: Callable[[], float], 
                     p:  [float], 
                     dp: [float], 
                     threshold: float,
        ) -> list:
    """
    Description
    -----------
        twiddle search optimizer

    Parameters
    ----------
    func: Callable
        the objective/loss function designed for a minimization optimization 
        routine 
        -- the lower of the returned value, the better the parameters --

    p: list
        initial parameter vector
    dp: list
        initial search step for each parameter
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


if __name__ == "__main__":
    import numpy as np

    # Twiddle search example
    #   f(x,y) = sin(x)^2 - cos(y)^2
    #   with initial guess of x=1, y=1:
    #       f_min = f(x=0, y=0) = -1
    #
    # >> python optimizer.py
    # [2.6463814045929285e-05, 2.6463814045929285e-05]
    #
    func = lambda p: np.sin(p[0])**2 - np.cos(p[1])**2
    print(twiddle_optimize(
                func,
                [1, 1],
                [0.1, 0.1],
                1e-4,
            )
    )
