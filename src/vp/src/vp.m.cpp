#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
    struct [[nodiscard]] queue_family_indices
    {
        std::optional<uint32_t> graphics_family;
    };

    queue_family_indices find_queue_families(VkPhysicalDevice device)
    {
        queue_family_indices indices;

        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families{count};
        vkGetPhysicalDeviceQueueFamilyProperties(device,
            &count,
            queue_families.data());

        uint32_t i = 0;
        for (auto const& queue_family : queue_families)
        {
            if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphics_family = i;
            }

            i++;
        }

        return indices;
    }

    [[nodiscard]] bool is_device_suitable(VkPhysicalDevice device)
    {
        auto const indices{find_queue_families(device)};

        return indices.graphics_family.has_value();
    }
} // namespace

class hello_triangle_application
{
public:
    void run()
    {
        init_window();
        init_vulkan();
        mainLoop();
        cleanup();
    }

private:
    void init_window()
    {
        constexpr int width = 800;
        constexpr int height = 600;
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_.reset(
            glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr));
    }

    void init_vulkan()
    {
        create_instance();
        physical_device_ = pick_physical_device();
        create_logical_device();
    }

    void create_instance()
    {
        VkApplicationInfo const app_info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0};

        VkInstanceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info};

        enumerate_extensions();

        uint32_t glfw_extension_count;
        char const** const glfw_extensions{
            glfwGetRequiredInstanceExtensions(&glfw_extension_count)};
        create_info.enabledExtensionCount = glfw_extension_count;
        create_info.ppEnabledExtensionNames = glfw_extensions;
        create_info.enabledLayerCount = 0;

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void enumerate_extensions()
    {
        uint32_t count;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions{count};
        vkEnumerateInstanceExtensionProperties(nullptr,
            &count,
            extensions.data());

        std::cout << "Available extensions:\n";
        for (auto const& extension : extensions)
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    VkPhysicalDevice pick_physical_device()
    {
        uint32_t count;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);

        if (count == 0)
        {
            throw std::runtime_error{
                "Failed to find GPUs with Vulkan support!"};
        }

        std::vector<VkPhysicalDevice> devices{count};
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        auto const device_it{std::ranges::find_if(devices, is_device_suitable)};
        if (device_it == devices.cend())
        {
            throw std::runtime_error{"failed to find a suitable GPU!"};
        }

        return *device_it;
    }

    void create_logical_device()
    {
        queue_family_indices indices = find_queue_families(physical_device_);

        float const priority{1.0f};
        VkDeviceQueueCreateInfo queue_create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = indices.graphics_family.value(),
            .queueCount = 1,
            .pQueuePriorities = &priority};

        VkPhysicalDeviceFeatures device_features{};

        VkDeviceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledLayerCount = 0,
            .enabledExtensionCount = 0,
            .pEnabledFeatures = &device_features};

        if (auto r = vkCreateDevice(physical_device_, &create_info, nullptr, &device_); r !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create logical device!"};
        }

        vkGetDeviceQueue(device_,
            indices.graphics_family.value(),
            0,
            &graphics_queue_);
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window_.get()))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        window_.reset();
        glfwTerminate();
    }

private:
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_{nullptr,
        glfwDestroyWindow};
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue graphics_queue_;
};

int main()
{
    hello_triangle_application app;

    try
    {
        app.run();
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
