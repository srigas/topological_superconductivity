# TSC Code

This repository contains python code for the purposes of TSC (topological superconductivity) simulations. Essentially, it is the "translated" version of an older FORTRAN codebase into Python.

## Motivation

While FORTRAN is an excellent programming language for scientific computing due to its speed, Python's numerous libraries can help optimize the code to such an extent that it surpasses FORTRAN's amazing benchmarks. This is exactly what we did here: we used NumPy to vectorize everything in the FORTRAN codebase (vectorized operations essentially correspond to optimized C code under the hood). For the operations that could not be vectorized (for example, due to memory limitations), we used the `numba.njit` module to compile the corresponding functions into C code, bypassing Python's interpreter. Finally, we applied parallelization using `joblib` to loops that were very time consuming even for the FORTRAN codebase. As a result, these optimizations led to an overall decrease of compute time, even when compared to FORTRAN using ifort with an Intel CPU.

## Running the code

The main codebase corresponds to the `tsc` python module. A user can either install the module locally, or simply keep the folder and work at the same or a different directory. The `docs` folder provides many examples of how to use the code. Especially the [Documentation](/docs/[0] Documentation) notebook contains a thorough description of all aspects of the codebase, its functions, the Theory behind them, as well as why we made specific choices when writing the code. For topics that are even more specialized (e.g. Parallelization), there are additional notebooks inside docs.

As you may have noticed, almost all experiments are run within .ipynb notebooks, with a `config.py` file at the same directory and usually with a `plotting.py` file to handle all plots. We advise you to keep this same structure when running your own experiments or tests.