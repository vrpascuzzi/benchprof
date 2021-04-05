// test_malloc.cc
#include <iostream>
#include <CL/sycl.hpp>

/*
-----------
CUDA
-----------
clang++ \
  -DSYCL_TARGET_CUDA \
  -Wno-unknown-cuda-version
  -fsycl
  -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
  -fsycl-unnamed-lambda
  -o test_malloc \
  test_malloc.cc 

-----------
HIP
-----------
syclcc -O2 \
  -DSYCL_TARGET_AMD \
  -Wno-ignored-attributes \
  --hipsycl-targets=hip:gfx900,gfx906,gfx908 \
  -o test_malloc \
  test_malloc.cc
*/

#ifdef SYCL_TARGET_CUDA
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& device) const override {
    const std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
    const std::string device_driver =
        device.get_info<cl::sycl::info::device::driver_version>();
    const std::string device_name = device.get_info<cl::sycl::info::device::name>();

    if (device.is_gpu() && (device_vendor.find("NVIDIA") != std::string::npos)) {
      return 1;
    }
    return -1;
  }
};
#endif

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
  CUDASelector selector;
  try {
    dev = cl::sycl::device(selector);
  } catch (...) {
  }
#elif defined SYCL_TARGET_AMD
  AMDSelector selector;
  try {
    dev = cl::sycl::device(selector);
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

// Gets the target device, as defined by the build configuration.
static inline cl::sycl::device GetTargetDevice() {
  cl::sycl::device dev;
  AMDSelector selector;
  try {
    dev = cl::sycl::device(selector);
  } catch (...) {
  }
  return dev;
}

// main
int main(int argc, char** argv) {
  // Choose device to run on and create queue
  cl::sycl::device device = GetTargetDevice();
  cl::sycl::queue queue(device);
  cl::sycl::context ctx(queue.get_context());

  // Get some device information and print to screen
  unsigned int vendor_id = static_cast<unsigned int>(
      queue.get_device().get_info<cl::sycl::info::device::vendor_id>());
  std::cout << "SYCL device: "
            << device.get_info<cl::sycl::info::device::name>()
            << "\nDriver version: "
            << device.get_info<cl::sycl::info::device::driver_version>()
            << "\nVendor ID: " << vendor_id << std::endl;

  // Allocate memory for an int on host and device
  int* host_int = (int*)malloc(sizeof(int));
  int* device_int = (int*)cl::sycl::malloc_device(sizeof(int), queue);

  // Initialize host int
  host_int[0] = 0;

  // Show initial data
  std::cout << "Initial host_int[0]: " << host_int[0] << std::endl;

  // Copy host data to device
  queue.memcpy(device_int, host_int, sizeof(int));

  // Modify the data on-device
  auto ev = queue.submit([&](cl::sycl::handler& h) {
    h.single_task([=]() { device_int[0] += 10; });
  });
  ev.wait();

  // Copy device data back to host
  queue.memcpy(host_int, device_int, sizeof(int)).wait();

  // Show result
  std::cout << "Result host_int[0]: " << host_int[0] << std::endl;

  // Free allocations
  cl::sycl::free(device_int, ctx);
  free(host_int);
  return 0;
}
