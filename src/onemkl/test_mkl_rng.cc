// test_mkl.cc
//
/*[CUDA]
 clang++ \
   -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
   -lonemkl -DSYCL_TARGET_CUDA=ON test_mkl_rng.cc
 [CPU]
 clang++ \
   -fsycl -lonemkl -DSYCL_TARGET_CPU=ON test_mkl_rng.cc \
   -o test_mkl_cpu.exe
 [GPU]
 clang++ \
   -fsycl -lonemkl -DSYCL_TARGET_GPU=ON test_mkl_rng.cc \
   -o test_mkl_gpu.exe
   */
/*
rm timing_curand_tpb512.csv ; \
for size in 1 10 100 10000 100000 1000000 10000000 100000000; do \
for name in "uniform_float" "uniform_double" "uniform_float_accurate" \
"uniform_double_accurate" "gaussian_float" "gaussian_double" "lognormal_float" \
"bits_int" "uniform_int" ; do \
./test_mkl_rng_cuda.exe 100 ${size} ${name} >> timing_mkl_rng_cuda.csv; \
done; \
done;

rm timing_mkl_rng_gpu.csv ; \
for size in 1 10 100 10000 100000 1000000 10000000 100000000; do \
for name in "uniform_float" "uniform_double" "uniform_float_accurate" \
"uniform_double_accurate" "gaussian_float" "gaussian_double" "lognormal_float" \
"bits_int" "uniform_int" ; do \
./test_mkl_rng_gpu.exe 100 ${size} ${name} >> timing_mkl_rng_gpu.csv; \
done; \
done;

rm timing_mkl_rng_cpu.csv ; \
for size in 1 10 100 10000 100000 1000000 10000000 100000000; do \
for name in "uniform_float" "uniform_double" "uniform_float_accurate" \
"uniform_double_accurate" "gaussian_float" "gaussian_double" "lognormal_float" \
"bits_int" "uniform_int" ; do \
./test_mkl_rng_cpu.exe 100 ${size} ${name} >> timing_mkl_rng_cpu.csv; \
done; \
done;
*/

#include <math.h>
#include <unistd.h>

#include <CL/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <vector>

#define UNIFORM_ARGS_FLOAT -1.0f, 5.0f
#define UNIFORM_ARGS_DOUBLE -1.0, 5.0
#define UNIFORM_ARGS_INT -1, 5

#define GAUSSIAN_ARGS_FLOAT -1.0f, 5.0f
#define GAUSSIAN_ARGS_DOUBLE -1.0, 5.0

#define LOGNORMAL_ARGS_FLOAT -1.0f, 5.0f, 1.0f, 2.0f
#define LOGNORMAL_ARGS_DOUBLE -1.0, 5.0, 1.0, 2.0

#define BERNOULLI_ARGS 0.5f

#define POISSON_ARGS 0.5

// Value to initialize random number generator
#define SEED 123456

// Value of Pi with many exact digits to compare with estimated value of Pi
#define PI 3.1415926535897932384626433832795

#ifdef SYCL_TARGET_CUDA
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor =
        device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name =
        device.get_info<cl::sycl::info::device::name>();

    if (device.is_gpu() &&
        (device_vendor.find("NVIDIA") != std::string::npos) &&
        (device_driver.find("CUDA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};
#endif

#ifdef SYCL_TARGET_HIP
class AMDSelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor =
        device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name =
        device.get_info<cl::sycl::info::device::name>();

    if (device.is_gpu() && (device_vendor.find("AMD") != std::string::npos)) {
      return 1;
    }
    return -1;
  }
};
#endif

// Gets the target device, as defined by the build configuration.
static inline cl::sycl::device GetTargetDevice() {
  cl::sycl::device dev;
#ifdef SYCL_TARGET_CUDA
  CUDASelector cuda_selector;
  try {
    dev = cl::sycl::device(cuda_selector);
  } catch (...) {
  }
#elif SYCL_TARGET_HIP
  AMDSelector amd_selector;
  try {
    dev = cl::sycl::device(amd_selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_DEFAULT
  dev = cl::sycl::device(cl::sycl::default_selector());
#elif defined SYCL_TARGET_CPU
  dev = cl::sycl::device(cl::sycl::cpu_selector());
#elif defined SYCL_TARGET_GPU
  dev = cl::sycl::device(cl::sycl::gpu_selector());
#else
  dev = cl::sycl::device(cl::sycl::host_selector());
#endif

  return dev;
}

template <typename Engine, typename Distr, typename... Args>
void test_rng(std::string name, cl::sycl::queue queue, size_t n_iters,
              std::int64_t n_points, Args... args) {
  using Type = typename Distr::result_type;

  // Clocks
  std::chrono::time_point<std::chrono::high_resolution_clock> start_tot;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_tot;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_kernel;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::vector<std::chrono::duration<double>> times_vec;
  std::vector<std::chrono::duration<double>> kernel_times_vec;
  std::chrono::duration<double> tot_time;
  std::chrono::duration<double> kernel_tot_time;
  std::chrono::duration<double> mean_time;
  std::chrono::duration<double> kernel_mean_time;
  double tot_var = 0.0;
  double kernel_var = 0.0;
  double tot_stddev = 0.0;
  double kernel_stddev = 0.0;

  for (unsigned int i = 0; i < n_iters; ++i) {
    start_tot = std::chrono::high_resolution_clock::now();
#ifdef SYCL_USE_USM
    // Create usm allocator
#ifdef USE_RT_API
    cl::sycl::usm_allocator<Type, cl::sycl::usm::alloc::shared, 64> allocator(
        queue);
#else
    cl::sycl::usm_allocator<Type, cl::sycl::usm::alloc::shared, 64> allocator(
        queue.get_context(), queue.get_device());
#endif

    // Allocate storage for random numbers
    std::vector<Type, decltype(allocator)> x(n_points, allocator);
#else
    cl::sycl::buffer<Type, 1> x(n_points);
#endif

    try {
      // Generator initialization
      Engine engine(queue, SEED);
      Distr distr(args...);

      start_kernel = std::chrono::high_resolution_clock::now();
#ifdef SYCL_USE_USM
      auto e = oneapi::mkl::rng::generate(distr, engine, n_points, x.data());
      // wait to finish generation
      e.wait();
#else
      oneapi::mkl::rng::generate(distr, engine, n_points, x);
#endif
      end = std::chrono::high_resolution_clock::now();
      kernel_times_vec.push_back(end - start_kernel);
    } catch (cl::sycl::exception const& e) {
      std::cout << "\t\tSYCL exception \n" << e.what() << std::endl;
    }

    // Print first few numbers
#ifndef SYCL_USE_USM
    auto acc = x.template get_access<cl::sycl::access::mode::read>();
#endif
    // for (size_t i = 0; i < 10; ++i) {
    //  std::cout << acc[i] << std::endl;
    //}
    end_tot = std::chrono::high_resolution_clock::now();
    times_vec.push_back(end_tot - start_tot);
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
    var += (t.count() - kernel_mean_time.count()) *
           (t.count() - kernel_mean_time.count());
  }
  var /= sizeof(kernel_times_vec);
  kernel_stddev = std::sqrt(var);
  // Print
  std::cout << name << "," << n_iters << "," << n_points << ","
            << tot_time.count() << "," << mean_time.count() << "," << tot_stddev
            << "," << kernel_tot_time.count() << "," << kernel_mean_time.count()
            << "," << kernel_stddev << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout
        << "useage: test_mkl_rng.exe <num_batches> <batch_size> <distr_type>\n";
    return 0;
  }

  size_t n_iters = atoi(argv[1]);
  size_t n_points = atoi(argv[2]);
  std::string name = std::string(argv[3]);

  auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  // Choose device to run on and create queue
  cl::sycl::device dev = GetTargetDevice();
  cl::sycl::queue queue(dev, exception_handler);
  std::string device_name =
      queue.get_device().get_info<cl::sycl::info::device::name>();
  // std::cout << "dev_name: " << device_name << std::endl;

  // Initialize output file
  std::string device_type;
  if (std::string(argv[0]).find("cpu") != std::string::npos) {
    device_type = "cpu";
  } else if (std::string(argv[0]).find("gpu") != std::string::npos) {
    device_type = "gpu";
  } else if (std::string(argv[0]).find("curand") != std::string::npos) {
    device_type = "curand";
  } else {
    std::cout << "unknown device\n";
    return 0;
  }

#ifdef USE_PHILOX
  using engine = oneapi::mkl::rng::philox4x32x10;
#elif defined USE_MRG
  using engine = oneapi::mkl::rng::mrg32k3a;
#endif

  if (name == "uniform_float") {
    std::string name = "uniform_float";
    test_rng<engine, oneapi::mkl::rng::uniform<
                         float, oneapi::mkl::rng::uniform_method::standard>>(
        name, queue, n_iters, n_points, UNIFORM_ARGS_FLOAT);
  } else if (name == "uniform_double") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::uniform<
                 double, oneapi::mkl::rng::uniform_method::standard>>(
        name, queue, n_iters, n_points, UNIFORM_ARGS_DOUBLE);
  } else if (name == "uniform_int") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::uniform<
                 std::int32_t, oneapi::mkl::rng::uniform_method::standard>>(
        name, queue, n_iters, n_points, UNIFORM_ARGS_DOUBLE);
  } else if (name == "uniform_float_accurate") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::uniform<
                 float, oneapi::mkl::rng::uniform_method::accurate>>(
        name, queue, n_iters, n_points, UNIFORM_ARGS_FLOAT);
  } else if (name == "uniform_double_accurate") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::uniform<
                 double, oneapi::mkl::rng::uniform_method::accurate>>(
        name, queue, n_iters, n_points, UNIFORM_ARGS_DOUBLE);
  } else if (name == "gaussian_float") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::gaussian<
                 float, oneapi::mkl::rng::gaussian_method::box_muller2>>(
        name, queue, n_iters, n_points, GAUSSIAN_ARGS_FLOAT);
  } else if (name == "gaussian_double") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::gaussian<
                 double, oneapi::mkl::rng::gaussian_method::box_muller2>>(
        name, queue, n_iters, n_points, GAUSSIAN_ARGS_DOUBLE);
  } else if (name == "lognormal_float") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::lognormal<
                 float, oneapi::mkl::rng::lognormal_method::box_muller2>>(
        name, queue, n_iters, n_points, LOGNORMAL_ARGS_FLOAT);
  } else if (name == "bits_int") {
    test_rng<oneapi::mkl::rng::philox4x32x10,
             oneapi::mkl::rng::bits<std::uint32_t>>(name, queue, n_iters,
                                                    n_points);
  } else {
    std::cout << "invalid distr_type\n";
  }
  return 0;
}
