//This program uses the host CURAND API.

/*
rm timing_curand_tpb512.csv
for size in 1 10 100 10000 100000 1000000 10000000 100000000; do \
for name in "uniform_float" "uniform_double" "uniform_float_accurate" \
"uniform_double_accurate" "gaussian_float" "gaussian_double" "lognormal_float" \
"bits_int" "uniform_int" ; do \
./test_curand_philox.exe 100 ${size} ${name} >> timing_curand_tpb512.csv; \
done; \
done;
 */
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include <curand.h>
#include <unistd.h>


#define THREADS_PER_BLOCK 256

#define UNIFORM_ARGS_FLOAT  -1.0f, 5.0f
#define UNIFORM_ARGS_DOUBLE -1.0, 5.0
#define UNIFORM_ARGS_INT    -1, 5

#define GAUSSIAN_ARGS_FLOAT  -1.0f, 5.0f
#define GAUSSIAN_ARGS_DOUBLE -1.0, 5.0

#define LOGNORMAL_ARGS_FLOAT  -1.0f, 5.0f, 1.0f, 2.0f
#define LOGNORMAL_ARGS_DOUBLE -1.0, 5.0, 1.0, 2.0

#define BERNOULLI_ARGS 0.5f

#define POISSON_ARGS 0.5

// Value to initialize random number generator
#define SEED 123456ULL

#define CUDA_CALL(x) do { \
    cudaError_t err = x; \
    if (err!=cudaSuccess) { \
        printf("Error %d at %s:%d\n",err,__FILE__,__LINE__);\
        return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { \
    curandStatus_t err = x; \
    if (err !=CURAND_STATUS_SUCCESS) { \
        printf("Error %d at %s:%d\n",err,__FILE__,__LINE__);\
        return EXIT_FAILURE;}} while(0)

template<typename Type>
__global__ void range_transform(std::uint64_t n, Type* devData, Type a, Type b) {
    int tid =  threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        devData[tid] = devData[tid] * (b - a) + a;
    }
}

template<typename Type>
__global__ void range_transform_int(std::uint64_t n, std::uint32_t* uniformData, Type* devData, Type a, Type b) {
    int tid =  threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        devData[tid] = a + uniformData[tid] % (b - a);
    }
}

int generate(size_t n_iters, size_t n_points, std::string name = "") {

  // Clocks
  std::chrono::time_point<std::chrono::high_resolution_clock> start_tot;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_tot;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_kernel;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_kernel;
  std::vector<std::chrono::duration<double>> times_vec;
  std::vector<std::chrono::duration<double>> kernel_times_vec;
  std::chrono::duration<double> tot_time;
  std::chrono::duration<double> kernel_tot_time;
  std::chrono::duration<double> mean_time;
  std::chrono::duration<double> kernel_mean_time;
  double var = 0.0;
  double tot_stddev = 0.0;
  double kernel_stddev = 0.0;


    if (name == "uniform_float") {
        using Type = float;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
            start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

    unsigned int nblocks = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerateUniform(gen, devData, n_points));
            range_transform<Type><<<nblocks,THREADS_PER_BLOCK>>>(n_points, devData, UNIFORM_ARGS_FLOAT);
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
    } else if (name == "uniform_double") {
        using Type = double;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
            start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

    unsigned int nblocks = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerateUniformDouble(gen, devData, n_points));
            // cudaDeviceSynchronize();
            range_transform<Type><<<nblocks,THREADS_PER_BLOCK>>>(n_points, devData, UNIFORM_ARGS_DOUBLE);
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            CURAND_CALL(curandDestroyGenerator(gen));
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
        }
     } else if (name == "gaussian_float") {
        using Type = float;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
             start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerateNormal(gen, devData, n_points, GAUSSIAN_ARGS_FLOAT));
            // cudaDeviceSynchronize();
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
     } else if (name == "gaussian_double") {
        using Type = double;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
             start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerateNormalDouble(gen, devData, n_points, GAUSSIAN_ARGS_DOUBLE));
            // cudaDeviceSynchronize();
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
    }  else if (name == "lognormal_float") {
        using Type = float;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
             start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

    unsigned int nblocks = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerateLogNormal(gen, devData, n_points, 0.0f, 1.0f));
            range_transform<Type><<<nblocks,THREADS_PER_BLOCK>>>(n_points, devData, -1.0f, 5.0f);
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
     } else if (name == "bits_int") {
        using Type = std::uint32_t;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
             start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

            /* Generate n floats on device */
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerate(gen, devData, n_points));
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
     } else if (name == "uniform_int") {
        using Type = std::int32_t;
        
        for (unsigned int i = 0; i < n_iters; ++i) {
             start_tot = std::chrono::high_resolution_clock::now();

        curandGenerator_t gen;
    /* Create pseudo-random number generator */
#ifdef USE_PHILOX
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_PHILOX4_32_10));
#elif defined USE_MRG
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#else
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_MRG32K3A));
#endif
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                SEED));

    unsigned int nblocks = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            /* Generate n floats on device */
            std::uint32_t* tempData;
            Type *devData, *hostData;
            /* Allocate n floats on host */
            hostData = (Type *)calloc(n_points, sizeof(Type));
            /* Allocate n floats on device */
            CUDA_CALL(cudaMalloc((void **)&tempData, n_points*sizeof(Type)));
            CUDA_CALL(cudaMalloc((void **)&devData, n_points*sizeof(Type)));
            start_kernel = std::chrono::high_resolution_clock::now();
            CURAND_CALL(curandGenerate(gen, tempData, n_points));
            cudaDeviceSynchronize();
            range_transform_int<Type><<<nblocks,THREADS_PER_BLOCK>>>(n_points, tempData, devData, -1, 5);
            cudaError_t status = cudaDeviceSynchronize();
            end_kernel = std::chrono::high_resolution_clock::now();
            kernel_times_vec.push_back(end_kernel - start_kernel);
        
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostData, devData, n_points * sizeof(float),
            cudaMemcpyDeviceToHost));

            /* Show result */
            // for(unsigned int i = 0; i < 10; i++) {
            //     printf("-- %u: %1.4f ", i, hostData[i]);
            // }
            // printf("\n");

            /* Cleanup */
            CUDA_CALL(cudaFree(devData));
            free(hostData);
            end_tot = std::chrono::high_resolution_clock::now();
            times_vec.push_back(end_tot - start_tot);
            CURAND_CALL(curandDestroyGenerator(gen));
        }
    } else {
        std::cout << "unknown distr_type\n";
        return EXIT_SUCCESS;
    }

  // Get timings
  for (auto t : times_vec) {
    tot_time += t;
  }
  for (auto t : kernel_times_vec) {
    kernel_tot_time += t;
  }

  mean_time = tot_time / float(n_iters);
  kernel_mean_time = kernel_tot_time / float(n_iters);
  for (auto t : times_vec) {
    var += (t.count() - mean_time.count()) * (t.count() - mean_time.count());
  }
  var /= sizeof(times_vec);
  tot_stddev = std::sqrt(var);

  var = 0.0;
  for (auto t : kernel_times_vec) {
    var += (t.count() - kernel_mean_time.count()) * (t.count() - kernel_mean_time.count());
  }
  var /= sizeof(kernel_times_vec);
  kernel_stddev = std::sqrt(var);
  // Print
  std::cout << name << ","
           << n_iters << ","
           << n_points << ","
           << tot_time.count() << ","
           << mean_time.count() << ","
           << tot_stddev << ","
           << kernel_tot_time.count() << ","
           << kernel_mean_time.count() << ","
           << kernel_stddev << std::endl;
           
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "useage: test_mkl_rng.exe <num_batches> <batch_size> <distr_type>\n";
        return 0;
    }
    size_t n_iters = atoi(argv[1]);
    size_t n_points = atoi(argv[2]);
    std::string name = std::string(argv[3]);
    std::vector<std::string> names = {"uniform_float", "uniform_double",
        "uniform_float_accurate", "uniform_double_accurate",
        "gaussian_float", "gaussian_double", "lognormal_float",
        "bits_int", "uniform_int"};
    // for (auto name : names) {
    //     generate(n_iters, n_points, name);
    // }
    if ((name == "gaussian_float" || name == "gaussian_double"
        || name == "lognormal_float") && n_points == 1)
        n_points = 2;
    generate(n_iters, n_points, name);

    return EXIT_SUCCESS;
}
