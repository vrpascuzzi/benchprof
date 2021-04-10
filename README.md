# benchprof
Repo for benchmarking and profiling code

## Reproducing SC21 paper results
1. Checkout this repo
2. Make sure to have the following:
    1. [Intel LLVM `sycl-nightly/20200330`](https://github.com/intel/llvm/tree/sycl-nightly/20210330)
    2. CUDA 10.2
    3. Nsight Compute >= 2020.2
    4. [hipSYCL 0.9.0](http://repo.urz.uni-heidelberg.de/sycl/test-plugin/rpm/centos7/) and dependencies (core, base, etc.)
    6. python3+matplotlib (for plots)
3. Then execute the following:
```
$ cd benchprof
$ export BENCHPROF_DIR=$PWD
$ cd onemkl
$ source scripts/rng_compile.sh <backend> # backend = host, intelcpu, intelgpu, \
                                          # mkl_curand, mkl_hiprand,            \
                                          # curand, hiprand
$ source scripts/rng_run.sh <backend>
$ python3 python/plot_clock_csv.py
```
Please report any issues with reproducibility.

The [FastCaloSim](https://github.com/vrpascuzzi/FastCaloSim-GPU/tree/benchmarking) input files are proprietary ATLAS Experiment data and so cannot be shared publicly.
As a result, reproducing our results for this application is not possible.
