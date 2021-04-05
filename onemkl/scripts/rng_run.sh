declare -a BATCHSIZES=(1 10 100 10000 100000 1000000 10000000 100000000) ;
declare -a DISTS=("uniform_float" "uniform_double" "gaussian_float" \
    "gaussian_double" "lognormal_float" "bits_int") ;

BACKEND=${1}

if [ "${BACKEND}" = "cpu" ]; then
    OUTCSV_BUFFER_PHILOX="./timing/timing_mkl_rng_cpu_buffer_philox.csv"
    OUTCSV_USM_PHILOX="./timing/timing_mkl_rng_cpu_usm_philox.csv"
    EXE_BUFFER_PHILOX=./exe/test_mkl_rng_cpu_buffer_philox.exe
    EXE_USM_PHILOX=./exe/test_mkl_rng_cpu_usm_philox.exe

    OUTCSV_BUFFER_MRG="./timing/timing_mkl_rng_cpu_buffer_mrg.csv"
    OUTCSV_USM_MRG="./timing/timing_mkl_rng_cpu_usm_mrg.csv"
    EXE_BUFFER_MRG=./exe/test_mkl_rng_cpu_buffer_mrg.exe
    EXE_USM_MRG=./exe/test_mkl_rng_cpu_usm_mrg.exe
elif [ "${BACKEND}" = "gpu" ]; then
    OUTCSV_BUFFER_PHILOX="./timing/timing_mkl_rng_gpu_buffer_philox.csv"
    OUTCSV_USM_PHILOX="./timing/timing_mkl_rng_gpu_usm_philox.csv"
    EXE_BUFFER_PHILOX=./exe/test_mkl_rng_gpu_buffer_philox.exe
    EXE_USM_PHILOX=./exe/test_mkl_rng_gpu_usm_philox.exe

    OUTCSV_BUFFER_MRG="./timing/timing_mkl_rng_gpu_buffer_mrg.csv"
    OUTCSV_USM_MRG="./timing/timing_mkl_rng_gpu_usm_mrg.csv"
    EXE_BUFFER_MRG=./exe/test_mkl_rng_gpu_buffer_mrg.exe
    EXE_USM_MRG=./exe/test_mkl_rng_gpu_usm_mrg.exe
elif [ "${BACKEND}" = "mkl_curand" ]; then
    OUTCSV_BUFFER_PHILOX="./timing/timing_mkl_rng_curand_buffer_philox.csv"
    OUTCSV_USM_PHILOX="./timing/timing_mkl_rng_curand_usm_philox.csv"
    EXE_BUFFER_PHILOX=./exe/test_mkl_rng_curand_buffer_philox.exe
    EXE_USM_PHILOX=./exe/test_mkl_rng_curand_usm_philox.exe

    OUTCSV_BUFFER_MRG="./timing/timing_mkl_rng_curand_buffer_mrg.csv"
    OUTCSV_USM_MRG="./timing/timing_mkl_rng_curand_usm_mrg.csv"
    EXE_BUFFER_MRG=./exe/test_mkl_rng_curand_buffer_mrg.exe
    EXE_USM_MRG=./exe/test_mkl_rng_curand_usm_mrg.exe
elif [ "${BACKEND}" = "cuda_curand" ]; then
    OUTCSV_USM_PHILOX="./timing/timing_cuda_curand_philox.csv"
    EXE_USM_PHILOX=./exe/test_cuda_curand_philox.exe

    OUTCSV_USM_MRG="./timing/timing_cuda_curand_mrg.csv"
    EXE_USM_MRG=./exe/test_cuda_curand_mrg.exe
else
    echo "unknown backend ${BACKEND}\n"
    echo "arg must be: cpu, gpu, curand_mkl, curand_cuda"
    exit 0
fi

################### BUFFER #####################
if [ -n ${EXE_BUFFER_PHILOX} ]; then
    rm ${OUTCSV_BUFFER_PHILOX} ;
    for size in ${BATCHSIZES[@]}; do
        for name in ${DISTS[@]}; do
             ${EXE_BUFFER_PHILOX} 100 ${size} ${name} >> ${OUTCSV_BUFFER_PHILOX};
        done;
    done;
fi
if [ -n ${EXE_BUFFER_MRG} ]; then
    rm ${OUTCSV_BUFFER_MRG} ;
    for size in ${BATCHSIZES[@]}; do
        for name in ${DISTS[@]}; do
             ${EXE_BUFFER_MRG} 100 ${size} ${name} >> ${OUTCSV_BUFFER_MRG};
        done;
    done;
fi
#################### USM #######################
if [ -n ${EXE_USM_PHILOX} ]; then
    rm ${OUTCSV_USM_PHILOX} ;
    for size in ${BATCHSIZES[@]}; do
        for name in ${DISTS[@]}; do
             ${EXE_USM_PHILOX} 100 ${size} ${name} >> ${OUTCSV_USM_PHILOX};
        done;
    done;
fi
if [ -n ${EXE_USM_MRG} ]; then
    rm ${OUTCSV_USM_MRG} ;
    for size in ${BATCHSIZES[@]}; do
        for name in ${DISTS[@]}; do
             ${EXE_USM_MRG} 100 ${size} ${name} >> ${OUTCSV_USM_MRG};
        done;
    done;
fi