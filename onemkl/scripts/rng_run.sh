#!/bin/bash

# Check if root directory is set; if not, fail
if [[ -z "${BENCHPROF_ROOT}" ]]; then
    echo "BENCHPROF_ROOT not set!"
    return
fi

# Check if build directory exists; if not, there's a problem.
export BLDDIR=${BENCHPROF_ROOT}/onemkl/bld
if [[ ! -d "${BLDDIR}" ]]; then
    echo "Build directory:"
    echo "  ${BLDDIR}"
    echo "does not exist! (Did you compile first?)"
    return
fi

# Check if a output directory exists; if not, create one
export OUTDIR=${BENCHPROF_ROOT}/onemkl/out
if [[ ! -d "${OUTDIR}" ]]; then
    echo "Output directory does not exist -- creating."
    mkdir ${OUTDIR}
fi

declare -a BATCHSIZES=(1 10 100 10000 100000 1000000 10000000 100000000) ;
declare -a DISTS=("uniform_float" "uniform_double" "gaussian_float" \
    "gaussian_double" "lognormal_float" "bits_int") ;

BACKEND=${1}

if [ "${BACKEND}" = "host" ]; then
    OUTCSV_BUFFER_PHILOX="${OUTDIR}/clock_mkl_rng_host_buffer_philox.csv"
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_mkl_rng_host_usm_philox.csv"
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_host_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_host_usm_philox.exe

    OUTCSV_BUFFER_MRG="${OUTDIR}/clock_mkl_rng_host_buffer_mrg.csv"
    OUTCSV_USM_MRG="${OUTDIR}/clock_mkl_rng_host_usm_mrg.csv"
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_host_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_host_usm_mrg.exe

elif [ "${BACKEND}" = "intelcpu" ]; then
    OUTCSV_BUFFER_PHILOX="${OUTDIR}/clock_mkl_rng_cpu_buffer_philox.csv"
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_mkl_rng_cpu_usm_philox.csv"
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_cpu_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_cpu_usm_philox.exe

    OUTCSV_BUFFER_MRG="${OUTDIR}/clock_mkl_rng_cpu_buffer_mrg.csv"
    OUTCSV_USM_MRG="${OUTDIR}/clock_mkl_rng_cpu_usm_mrg.csv"
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_cpu_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_cpu_usm_mrg.exe
elif [ "${BACKEND}" = "intelgpu" ]; then
    OUTCSV_BUFFER_PHILOX="${OUTDIR}/clock_mkl_rng_intelgpu_buffer_philox.csv"
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_mkl_rng_intelgpu_usm_philox.csv"
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_gpu_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_gpu_usm_philox.exe

    OUTCSV_BUFFER_MRG="${OUTDIR}/clock_mkl_rng_intelgpu_buffer_mrg.csv"
    OUTCSV_USM_MRG="${OUTDIR}/clock_mkl_rng_intelgpu_usm_mrg.csv"
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_gpu_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_gpu_usm_mrg.exe
elif [ "${BACKEND}" = "mkl_curand" ]; then
    OUTCSV_BUFFER_PHILOX="${OUTDIR}/clock_mkl_rng_curand_buffer_philox.csv"
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_mkl_rng_curand_usm_philox.csv"
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_curand_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_curand_usm_philox.exe

    OUTCSV_BUFFER_MRG="${OUTDIR}/clock_mkl_rng_curand_buffer_mrg.csv"
    OUTCSV_USM_MRG="${OUTDIR}/clock_mkl_rng_curand_usm_mrg.csv"
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_curand_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_curand_usm_mrg.exe
elif [ "${BACKEND}" = "mkl_hiprand" ]; then
    OUTCSV_BUFFER_PHILOX="${OUTDIR}/clock_mkl_rng_hiprand_buffer_philox.csv"
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_mkl_rng_hiprand_usm_philox.csv"
    EXE_BUFFER_PHILOX=${BLDDIR}/test_mkl_rng_hiprand_buffer_philox.exe
    EXE_USM_PHILOX=${BLDDIR}/test_mkl_rng_hiprand_usm_philox.exe

    OUTCSV_BUFFER_MRG="${OUTDIR}/clock_mkl_rng_hiprand_buffer_mrg.csv"
    OUTCSV_USM_MRG="${OUTDIR}/clock_mkl_rng_hiprand_usm_mrg.csv"
    EXE_BUFFER_MRG=${BLDDIR}/test_mkl_rng_hiprand_buffer_mrg.exe
    EXE_USM_MRG=${BLDDIR}/test_mkl_rng_hiprand_usm_mrg.exe
elif [ "${BACKEND}" = "curand" ]; then
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_curand_philox.csv"
    EXE_USM_PHILOX=${BLDDIR}/test_cuda_curand_philox.exe

    OUTCSV_USM_MRG="${OUTDIR}/clock_curand_mrg.csv"
    EXE_USM_MRG=${BLDDIR}/test_cuda_curand_mrg.exe
elif [ "${BACKEND}" = "hiprand" ]; then
    OUTCSV_USM_PHILOX="${OUTDIR}/clock_hiprand_philox.csv"
    EXE_USM_PHILOX=${BLDDIR}/test_hiprand_philox.exe

    OUTCSV_USM_MRG="${OUTDIR}/clock_hiprand_mrg.csv"
    EXE_USM_MRG=${BLDDIR}/test_hiprand_mrg.exe
else
    echo "unknown backend ${BACKEND}\n"
    echo "arg must be: intelcpu, intelgpu, mkl_curand, mkl_hiprand, curand, hiprand"
    return
fi

################### BUFFER #####################
if [ ! -z ${EXE_BUFFER_PHILOX} ]; then
    if [[ -f ${OUTCSV_BUFFER_PHILOX} ]]; then
        rm ${OUTCSV_BUFFER_PHILOX} ;
    fi
    echo "Executing PHILOX using Buffer API..."
    for size in ${BATCHSIZES[@]}; do
        echo "Batch size: ${size}"
        for name in ${DISTS[@]}; do
            echo "  Distribution: ${name}"
            ${EXE_BUFFER_PHILOX} 100 ${size} ${name} >> ${OUTCSV_BUFFER_PHILOX};
        done;
    done;
fi
if [ ! -z ${EXE_BUFFER_MRG} ]; then
    if [[ -f ${OUTCSV_BUFFER_PHILOX} ]]; then
        rm ${OUTCSV_BUFFER_MRG} ;
    fi
    echo "Executing MRG using Buffer API..."
    for size in ${BATCHSIZES[@]}; do
        echo "Batch size: ${size}"
        for name in ${DISTS[@]}; do
            echo "  Distribution: ${name}"
            ${EXE_BUFFER_MRG} 100 ${size} ${name} >> ${OUTCSV_BUFFER_MRG};
        done;
    done;
fi
#################### USM #######################
if [ ! -z ${EXE_USM_PHILOX} ]; then
    if [[ -f ${OUTCSV_USM_PHILOX} ]]; then
        rm ${OUTCSV_USM_PHILOX} ;
    fi
    echo "Executing PHILOX using USM API..."
    for size in ${BATCHSIZES[@]}; do
        echo "Batch size: ${size}"
        for name in ${DISTS[@]}; do
            echo "  Distribution: ${name}"
            ${EXE_USM_PHILOX} 100 ${size} ${name} >> ${OUTCSV_USM_PHILOX};
        done;
    done;
fi
if [ ! -z ${EXE_USM_MRG} ]; then
    if [[ -f ${OUTCSV_USM_MRG} ]]; then
        rm ${OUTCSV_USM_MRG} ;
    fi
    echo "Executing MRG using USM API..."
    for size in ${BATCHSIZES[@]}; do
        echo "Batch size: ${size}"
        for name in ${DISTS[@]}; do
            echo "  Distribution: ${name}"    
            ${EXE_USM_MRG} 100 ${size} ${name} >> ${OUTCSV_USM_MRG};
        done;
    done;
fi