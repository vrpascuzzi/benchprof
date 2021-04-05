BACKEND=${1}

if [ "${BACKEND}" = "cpu" ]; then
    EXE_BUFFER_PHILOX=../build/test_mkl_rng_cpu_buffer_philox.exe
    EXE_USM_PHILOX=../build/test_mkl_rng_cpu_usm_philox.exe
    EXE_BUFFER_MRG=../build/test_mkl_rng_cpu_buffer_mrg.exe
    EXE_USM_MRG=../build/test_mkl_rng_cpu_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_PHILOX=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_MRG=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CPU=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
elif [ "${BACKEND}" = "gpu" ]; then
    EXE_BUFFER_PHILOX=../build/test_mkl_rng_gpu_buffer_philox.exe
    EXE_USM_PHILOX=../build/test_mkl_rng_gpu_usm_philox.exe
    EXE_BUFFER_MRG=../build/test_mkl_rng_gpu_buffer_mrg.exe
    EXE_USM_MRG=../build/test_mkl_rng_gpu_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_PHILOX=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_MRG=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_GPU=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
elif [ "${BACKEND}" = "mkl_curand" ]; then
    EXE_BUFFER_PHILOX=../build/test_mkl_rng_curand_buffer_philox.exe
    EXE_USM_PHILOX=../build/test_mkl_rng_curand_usm_philox.exe
    EXE_BUFFER_MRG=../build/test_mkl_rng_curand_buffer_mrg.exe
    EXE_USM_MRG=../build/test_mkl_rng_curand_usm_mrg.exe
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_PHILOX=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_PHILOX} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_PHILOX=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_PHILOX}
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_MRG=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_BUFFER_MRG} &&
    clang++ \
        -fsycl \
        -lonemkl \
        -DSYCL_TARGET_CUDA=ON \
        -DUSE_MRG=ON \
        -DSYCL_USE_USM=ON \
        ../src/test_mkl_rng.cc \
        -o ${EXE_USM_MRG}
elif [ "${BACKEND}" = "cuda_curand" ]; then
    EXE_PHILOX=../build/test_cuda_curand_philox.exe
    EXE_MRG=../build/test_cuda_curand_mrg.exe
    nvcc -lcurand -arch=sm_75 ../src/test_curand.cu \
        -DUSE_PHILOX=ON -o ${EXE_PHILOX}
    nvcc -lcurand -arch=sm_75 ../src/test_curand.cu \
        -DUSE_MRG=ON -o ${EXE_MRG}
else
    echo "unknown backend ${BACKEND}\n"
    echo "arg must be: cpu, gpu, mkl_curand, cuda_curand"
    exit 0
fi
