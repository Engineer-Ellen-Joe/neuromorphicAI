#include <pybind11/pybind11.h>
#include "lve_window.hpp"
#include "lve_device.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vulkan_setup, m) {
    py::class_<lve::LveWindow>(m, "LveWindow")
        .def(py::init<int, int, std::string>());

    py::class_<lve::LveDevice>(m, "LveDevice")
        .def(py::init<lve::LveWindow &>())
        .def("device_handle", [](lve::LveDevice &device) {
            return reinterpret_cast<uint64_t>(device.device());
        })
        .def("physical_device_handle", [](lve::LveDevice &device) {
            return reinterpret_cast<uint64_t>(device.getPhysicalDevice());
        });
}
