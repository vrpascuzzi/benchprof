// test_atomic.cc

#define HIPSYCL_EXT_FP_ATOMICS
#include <CL/sycl.hpp>


#include <iostream>
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "="  VALUE(var)
/*
-----------
CUDA
-----------
clang++ \
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
  -Wno-ignored-attributes \
  --hipsycl-targets=hip:gfx900,gfx906,gfx908 \
  -o test_malloc \
  test_malloc.cc
*/

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
  // Choose device to run on and create queue
  cl::sycl::device device = GetTargetDevice();
  cl::sycl::queue queue(device);
  cl::sycl::context ctx(queue.get_context());

  // Get some device information and print to screen
  unsigned int vendor_id = static_cast<unsigned int>(
      queue.get_device().get_info<cl::sycl::info::device::vendor_id>());
  std::cout << "Running on \n"
            << "\tSYCL device: "
            << device.get_info<cl::sycl::info::device::name>()
            << "\n\tDriver version: "
            << device.get_info<cl::sycl::info::device::driver_version>()
            << "\n\tVendor ID: " << vendor_id << std::endl;

  // Allocate memory for an int on host and device
  std::vector<float> host_data(1);

  // Initialize host int
  host_data[0] = 1.1;

  // Show initial data
  std::cout << "Initial host_data[0]: " << host_data[0] << std::endl;

  // Create buffer from host data
  cl::sycl::buffer<float, 1> buf_host(host_data.data(), cl::sycl::range<1>(1));

  // Modify the data on-device
  auto ev = queue.submit([&](cl::sycl::handler& h) {
    auto acc = buf_host.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task([acc]() { 
    #ifdef SYCL_DEVICE_ONLY
    cl::sycl::atomic<float> ptr{acc.get_pointer()};
    ptr.fetch_add(3.5f);
    #endif
    });
  });
  ev.wait();

  // Show result data
  auto acc = buf_host.template get_access<cl::sycl::access::mode::read>();
  std::cout << "Result host_data[0]: " << host_data[0] << std::endl;

  return 0;
}
