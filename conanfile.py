from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout

require_conan_version = ">=2.0"

class VPConan(ConanFile):
    name = "vp"
    settings = "os", "compiler", "build_type", "arch"
    version = "0.1"

    exports_sources = "cmake", "src", "CMakeLists.txt", "LICENSE"

    def requirements(self):
        self.requires("vulkan-headers/1.3.268.0")
        self.requires("vulkan-loader/1.3.268.0")
        self.requires("glfw/3.4")
        self.requires("glm/cci.20230113")
        self.requires("stb/cci.20230920")
        self.requires("tinyobjloader/1.0.7")

    def build_requirements(self):
        self.tool_requires("cmake/[^3.27]")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = "ConanPresets.json"
        tc.generate()

        cmake = CMakeDeps(self)
        cmake.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

