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
  std::vector<int> host_int(1);

  // Initialize host int
  host_int[0] = 0;

  // Show initial data
  std::cout << "Initial host_int[0]: " << host_int[0] << std::endl;

  // Create buffer from host data
  cl::sycl::buffer<int, 1> buf_host(host_int.data(), cl::sycl::range<1>(1));

  // Modify the data on-device
  auto ev = queue.submit([&](cl::sycl::handler& h) {
    auto acc = buf_host.template get_access<cl::sycl::access::mode::atomic>(h);
    // auto acc = buf_host.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task([=]() { acc[0].fetch_add(1); });
    // h.single_task([=]() { acc[0] += 10; });
  });
  ev.wait();

  // Show result data
  auto acc = buf_host.template get_access<cl::sycl::access::mode::read>();
  std::cout << "Result host_int[0]: " << host_int[0] << std::endl;

  return 0;
}

