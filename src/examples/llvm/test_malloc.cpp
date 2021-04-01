#include <CL/sycl.hpp>
#include <iostream>

/*
clang++ -fsycl \
  -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
  -fsycl-unnamed-lambda \
  test_atomic.cpp
*/

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
        (device_vendor.find("NVIDIA") != std::string::npos)) {
      return 1;
    }
    return -1;
  }
};

// Gets the target device, as defined by the build configuration.
static inline cl::sycl::device GetTargetDevice() {
  cl::sycl::device dev;
  CUDASelector selector;
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
  std::cout << "Running on \n"
            << "\tSYCL device: "
            << device.get_info<cl::sycl::info::device::name>()
            << "\n\tDriver version: "
            << device.get_info<cl::sycl::info::device::driver_version>()
            << "\n\tVendor ID: " << vendor_id << std::endl;

  // Allocate memory for an int on host and device
  int* host_int = (int*)malloc(sizeof(int));
  int* device_int = (int*)cl::sycl::malloc_device(sizeof(int), queue);

  // Initialize host int
  host_int[0] = 1;

  // Copy host data to device
  queue.memcpy(device_int, host_int, sizeof(int)).wait();

  // Modify the data on-device
  auto ev = queue.submit([&](cl::sycl::handler& h) {
    h.single_task([=]() { device_int[0] += 10; });
  });
  ev.wait();

  // Copy device data back to hos
  queue.memcpy(host_int, device_int, sizeof(int)).wait();

  // Show result
  std::cout << "*host_int: " << *host_int << std::endl;

  // Modify the data on-device
  ev = queue.submit([&](cl::sycl::handler& h) {
    h.single_task([=]() { device_int[0] += 10; });
  });
  ev.wait();

  // Copy device data back to host
  queue.memcpy(host_int, device_int, sizeof(int)).wait();

  // Show result
  std::cout << "*host_int: " << *host_int << std::endl;

  // Free allocations
  cl::sycl::free(device_int, ctx);
  free(host_int);
  return 0;
}
