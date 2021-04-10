#!/bin/bash

# Check if root directory is set; if not, set it
if [[ -z "${BENCHPROF_ROOT}" ]]; then
    export BENCHPROF_ROOT=${PWD}../../
fi

# Check if a build directory exists; if not, create one
export BLDDIR=${BENCHPROF_ROOT}/onemkl/bld
if [[ ! -d "${BLDDIR}" ]]; then
    echo "Build directory does not exist -- creating."
    mkdir ${BLDDIR}
fi

export SRCDIR=${BENCHPROF_ROOT}/onemkl/src

# Get the selected back-end from command line
BACKEND=${1}

# intelcpu
if [ "${BACKEND}" = "host" ]; then
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_host_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_host_usm_philox.exe
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_host_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_host_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DUSE_PHILOX=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DUSE_MRG=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}

# intelcpu
elif [ "${BACKEND}" = "intelcpu" ]; then
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_cpu_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_cpu_usm_philox.exe
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_cpu_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_cpu_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_PHILOX=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_MRG=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
# intelgpu
elif [ "${BACKEND}" = "intelgpu" ]; then
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_gpu_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_gpu_usm_philox.exe
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_gpu_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_gpu_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_PHILOX=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_MRG=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
# mkl_curand
elif [ "${BACKEND}" = "mkl_curand" ]; then
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_curand_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_curand_usm_philox.exe
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_curand_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_curand_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_PHILOX=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_MRG=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
# mkl_hiprand
elif [ "${BACKEND}" = "mkl_hiprand" ]; then
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_hiprand_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_hiprand_usm_philox.exe
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_hiprand_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_hiprand_usm_mrg.exe
    syclcc \
        -O2 \
        -Wno-ignored-attributes \
        -fsycl \
        -lonemkl \
        --hipsycl-targets=hip:gfx900 \
        -DSYCL_TARGET_HIP=ON \
        -DUSE_PHILOX=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    syclcc \
        -O2 \
        -Wno-ignored-attributes \
        -fsycl \
        -lonemkl \
        --hipsycl-targets=hip:gfx900 \
        -DSYCL_TARGET_HIP=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    syclcc \
        -O2 \
        -Wno-ignored-attributes \
        -fsycl \
        -lonemkl \
        --hipsycl-targets=hip:gfx900 \
        -DSYCL_TARGET_HIP=ON \
        -DUSE_MRG=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    syclcc \
        -O2 \
        -Wno-ignored-attributes \
        -fsycl \
        -lonemkl \
        --hipsycl-targets=hip:gfx900 \
        -DSYCL_TARGET_HIP=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ${SRCDIR}/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
# curand
elif [ "${BACKEND}" = "curand" ]; then
    EXE_PHILOX=${BLDDIR}/test_cuda_curand_philox.exe
    EXE_MRG=${BLDDIR}/test_cuda_curand_mrg.exe
    nvcc -lcurand -arch=sm_75 ${SRCDIR}/test_curand.cu \
        -DUSE_PHILOX=ON -o ${EXE_PHILOX}
    nvcc -lcurand -arch=sm_75 ${SRCDIR}/test_curand.cu \
        -DUSE_MRG=ON -o ${EXE_MRG}
# hiprand
elif [ "${BACKEND}" = "hiprand" ]; then
    EXE_PHILOX=${BLDDIR}/test_hiprand_philox.exe
    EXE_MRG=${BLDDIR}/test_hiprand_mrg.exe
    hipcc \
        --gcc-toolchain=`which gcc | sed "s/\/bin\/gcc$//"` \
        --rocm-path=/opt/hipSYCL/rocm/hip \
        --rocm-device-lib-path=/opt/hipSYCL/rocm/rocm-device-libs/amdgcn/bitcode \
        -lhiprand \
        --offload-arch=gfx900 \
        -DUSE_PHILOX=ON -o ${EXE_PHILOX} \
        ${SRCDIR}/test_hiprand.cpp
    hipcc \
        --gcc-toolchain=`which gcc | sed "s/\/bin\/gcc$//"` \
        --rocm-path=/opt/hipSYCL/rocm/hip \
        --rocm-device-lib-path=/opt/hipSYCL/rocm/rocm-device-libs/amdgcn/bitcode \
        -lhiprand \
        --offload-arch=gfx900 \
        -DUSE_MRG=ON -o ${EXE_MRG} \
        ${SRCDIR}/test_hiprand.cpp
else
    echo "unknown backend ${BACKEND}\n"
    echo "arg must be: host, intelcpu, intelgpu, mkl_curand, mkl_hiprand, curand, hiprand"
    return
fi
