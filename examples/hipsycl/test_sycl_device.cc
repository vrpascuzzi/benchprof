// test_sycl_device.cc
#include <iostream>
#include <CL/sycl.hpp>

/*
/opt/hipSYCL/bin/syclcc \
  --hipsycl-targets=hip:gfx900,gfx906 \
  -I/opt/hipSYCL/rocm/hiprand/include \
  -I/opt/hipSYCL/rocm/rocrand/include \
  -L/opt/hipSYCL/rocm/hiprand/lib \
  -lhiprand \
  -L/opt/hipSYCL/rocm/rocrand/lib \
  -lrocrand \
  -DSYCL_TARGET_AMD \
  -o test_sycl_device test_sycl_device.cc



/opt/hipSYCL/bin/syclcc \
  --hipsycl-targets=hip:gfx900,gfx906 \
  -lonemkl -o test_mkl_rng test_mkl_rng.cc 
*/

#include <hiprand.h>

#define HIP_CALL(x) do { \
    hipError_t err = x; \
    if (err!=hipSuccess) { \
        printf("Error %d at %s:%d\n",err,__FILE__,__LINE__);\
        return EXIT_FAILURE;}} while(0)

#define HIPRAND_CALL( x )                                                                                               \
  if ( ( x ) != HIPRAND_STATUS_SUCCESS ) {                                                                              \
    printf( "Error at %s:%d\n", __FILE__, __LINE__ );                                                                  \
    exit( EXIT_FAILURE );                                                                                              \
  }


#ifdef SYCL_TARGET_AMD
class AMDSelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name = device.get_info<cl::sycl::info::device::name>();

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
#elif defined SYCL_TARGET_AMD
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

// main
int main(int argc, char** argv) {
  using Type = float;
  
  // Choose device to run on and create queue
  cl::sycl::device device = GetTargetDevice();
  cl::sycl::queue queue(device);
  unsigned int vendor_id = static_cast<unsigned int>(
      queue.get_device().get_info<cl::sycl::info::device::vendor_id>());
  std::cout << "Running on \n"
            << "\tSYCL device: " << device.get_info<cl::sycl::info::device::name>()
            << "\n\tDriver version: " << device.get_info<cl::sycl::info::device::driver_version>()
            << "\n\tVendor ID: " << vendor_id
            << std::endl;

  std::cout << "Setting up generator...\n";
  hiprandGenerator_t gen;
  HIPRAND_CALL(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_PHILOX4_32_10));
  HIPRAND_CALL(hiprandSetPseudoRandomGeneratorSeed(gen, 1234));

  std::cout << "Allocating memory...\n";
  unsigned int npoints = 1000;
  Type *devData, *hostData;
  /* Allocate n floats on host */
  hostData = (Type *)calloc(npoints, sizeof(Type));
  /* Allocate n floats on device */
  HIP_CALL(hipMalloc((void **)&devData, npoints*sizeof(Type)));
  
  std::cout << "Generating...\n";
  queue.submit([&](cl::sycl::handler &cgh) {
    cgh.hipSYCL_enqueue_custom_operation([=](cl::sycl::interop_handle &ih) {
      hiprandStatus_t status;
      HIPRAND_CALL(hiprandGenerateUniform(gen, devData, npoints));
    });
  });
  queue.wait();

  std::cout << "D2H...\n";
  HIP_CALL(hipMemcpy(hostData, devData, npoints * sizeof(float),
            hipMemcpyDeviceToHost));

  // Show results
  std::cout << "Results...\n";
  for (unsigned int i = 0; i < 10; i++) {
    std::cout << hostData[i] << " ";
  }
  std::cout << "\n";
  std::cout << "Freeing memory...\n";
  HIP_CALL(hipFree(devData));
  free(hostData);
  HIPRAND_CALL(hiprandDestroyGenerator(gen));
  return 0;
}
