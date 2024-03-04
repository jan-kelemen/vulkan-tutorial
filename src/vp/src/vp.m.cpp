#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    [[nodiscard]] std::vector<char> read_file(std::filesystem::path const& file)
    {
        std::ifstream stream{file, std::ios::ate | std::ios::binary};

        if (!stream)
        {
            throw std::runtime_error{"failed to open file!"};
        }

        auto const file_size{static_cast<size_t>(stream.tellg())};

        std::vector<char> buffer(file_size);
        stream.seekg(0);
        stream.read(buffer.data(), file_size);

        return buffer;
    }
} // namespace

namespace
{
    std::array<char const*, 1> const device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    struct [[nodiscard]] queue_family_indices
    {
        std::optional<uint32_t> graphics_family;
        std::optional<uint32_t> present_family;
    };

    queue_family_indices find_queue_families(VkPhysicalDevice device,
        VkSurfaceKHR surface)
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

            VkBool32 present_support;
            vkGetPhysicalDeviceSurfaceSupportKHR(device,
                i,
                surface,
                &present_support);
            if (present_support)
            {
                indices.present_family = i;
            }

            if (indices.graphics_family && indices.present_family)
            {
                break;
            }

            i++;
        }

        return indices;
    }

    struct [[nodiscard]] swap_chain_support_details
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> present_modes;
    };

    swap_chain_support_details query_swap_chain_support(VkPhysicalDevice device,
        VkSurfaceKHR surface)
    {
        swap_chain_support_details details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device,
            surface,
            &details.capabilities);

        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device,
            surface,
            &format_count,
            nullptr);

        if (format_count != 0)
        {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device,
                surface,
                &format_count,
                details.formats.data());
        }

        uint32_t present_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device,
            surface,
            &present_count,
            nullptr);

        if (present_count != 0)
        {
            details.present_modes.resize(present_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device,
                surface,
                &present_count,
                details.present_modes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR choose_swap_surface_format(
        std::span<VkSurfaceFormatKHR const> available_formats)
    {
        if (auto const it{std::ranges::find_if(available_formats,
                [](VkSurfaceFormatKHR const& f)
                {
                    return f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                })};
            it != available_formats.cend())
        {
            return *it;
        }

        return available_formats.front();
    }

    VkPresentModeKHR choose_swap_present_mode(
        std::span<VkPresentModeKHR const> available_present_modes)
    {
        constexpr auto preffered_mode{VK_PRESENT_MODE_MAILBOX_KHR};
        return std::ranges::find(available_present_modes, preffered_mode) !=
                available_present_modes.cend()
            ? preffered_mode
            : VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D choose_swap_extent(GLFWwindow* window,
        VkSurfaceCapabilitiesKHR const& capabilities)
    {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width;
        int height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actual_extent = {static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)};

        actual_extent.width = std::clamp(actual_extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actual_extent.height = std::clamp(actual_extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actual_extent;
    }

    [[nodiscard]] bool extensions_supported(VkPhysicalDevice device)
    {
        uint32_t count;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available_extensions{count};
        vkEnumerateDeviceExtensionProperties(device,
            nullptr,
            &count,
            available_extensions.data());

        std::set<std::string_view> required_extensions(
            device_extensions.cbegin(),
            device_extensions.cend());
        for (auto const& extension : available_extensions)
        {
            required_extensions.erase(extension.extensionName);
        }

        return required_extensions.empty();
    }

    [[nodiscard]] bool is_device_suitable(VkPhysicalDevice device,
        VkSurfaceKHR surface)
    {
        auto const indices{find_queue_families(device, surface)};

        bool swap_chain_adequate = false;
        if (extensions_supported(device))
        {
            auto const swap_chain_support{
                query_swap_chain_support(device, surface)};

            swap_chain_adequate = !swap_chain_support.formats.empty() &&
                !swap_chain_support.present_modes.empty();
        }

        return indices.graphics_family && indices.present_family &&
            swap_chain_adequate;
    }

    [[nodiscard]] VkShaderModule create_shader_module(VkDevice device,
        std::span<char const> code)
    {
        VkShaderModuleCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<uint32_t const*>(code.data())};

        VkShaderModule module;
        if (vkCreateShaderModule(device, &create_info, nullptr, &module) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create shader module"};
        }

        return module;
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
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swap_chain();
        create_image_views();
        create_render_pass();
        create_graphics_pipeline();
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

    void create_surface()
    {
        if (glfwCreateWindowSurface(instance_,
                window_.get(),
                nullptr,
                &surface_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface!");
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

    void pick_physical_device()
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

        auto const device_it{std::ranges::find_if(devices,
            [this](auto const& d)
            { return is_device_suitable(d, this->surface_); })};
        if (device_it == devices.cend())
        {
            throw std::runtime_error{"failed to find a suitable GPU!"};
        }

        physical_device_ = *device_it;
    }

    void create_logical_device()
    {
        queue_family_indices indices =
            find_queue_families(physical_device_, surface_);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_families = {indices.graphics_family.value(),
            indices.present_family.value()};
        float const priority{1.0f};
        for (uint32_t family : unique_families)
        {
            VkDeviceQueueCreateInfo queue_create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family,
                .queueCount = 1,
                .pQueuePriorities = &priority};

            queue_create_infos.push_back(queue_create_info);
        }

        VkPhysicalDeviceFeatures device_features{};

        VkDeviceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount =
                static_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledLayerCount = 0,
            .enabledExtensionCount =
                static_cast<uint32_t>(device_extensions.size()),
            .ppEnabledExtensionNames = device_extensions.data(),
            .pEnabledFeatures = &device_features};

        if (auto r = vkCreateDevice(physical_device_,
                &create_info,
                nullptr,
                &device_);
            r != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create logical device!"};
        }

        vkGetDeviceQueue(device_,
            indices.graphics_family.value(),
            0,
            &graphics_queue_);

        vkGetDeviceQueue(device_,
            indices.present_family.value(),
            0,
            &present_queue_);
    }

    void create_swap_chain()
    {
        swap_chain_support_details const support{
            query_swap_chain_support(physical_device_, surface_)};
        VkSurfaceFormatKHR const surface_format{
            choose_swap_surface_format(support.formats)};
        VkPresentModeKHR const present_mode{
            choose_swap_present_mode(support.present_modes)};
        VkExtent2D extent{
            choose_swap_extent(window_.get(), support.capabilities)};

        uint32_t image_count{support.capabilities.minImageCount + 1};
        if (support.capabilities.maxImageCount > 0)
        {
            image_count =
                std::min(support.capabilities.maxImageCount, image_count);
        }

        VkSwapchainCreateInfoKHR create_info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = support.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE};

        queue_family_indices const indices{
            find_queue_families(physical_device_, surface_)};
        std::array queue_family_indices{indices.present_family.value(),
            indices.present_family.value()};

        if (indices.graphics_family != indices.present_family)
        {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices.data();
        }
        else
        {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0;
            create_info.pQueueFamilyIndices = nullptr;
        }

        if (vkCreateSwapchainKHR(device_,
                &create_info,
                nullptr,
                &swap_chain_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create swap chain!"};
        }

        vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
        swap_chain_images_.resize(image_count);
        vkGetSwapchainImagesKHR(device_,
            swap_chain_,
            &image_count,
            swap_chain_images_.data());
        swap_chain_image_format_ = surface_format.format;
        swap_chain_extent_ = extent;
    }

    void create_image_views()
    {
        constexpr VkComponentMapping components{
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY};

        constexpr VkImageSubresourceRange subresource_range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1};

        swap_chain_image_views_.resize(swap_chain_images_.size());
        VkImageView* current_view{swap_chain_image_views_.data()};

        for (VkImage const image : swap_chain_images_)
        {
            VkImageViewCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swap_chain_image_format_,
                .components = components,
                .subresourceRange = subresource_range};

            if (vkCreateImageView(device_,
                    &create_info,
                    nullptr,
                    current_view++) != VK_SUCCESS)
            {
                throw std::runtime_error{"failed to create image views!"};
            }
        }
    }

    void create_render_pass()
    {
        VkAttachmentDescription color_attachment{
            .format = swap_chain_image_format_,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

        VkAttachmentReference color_attachment_ref{.attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_ref};

        VkRenderPassCreateInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &color_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass};

        if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create render pass"};
        }
    }

    void create_graphics_pipeline()
    {
        auto const vert_shader{read_file("../shaders/vert.spv")};
        auto const frag_shader{read_file("../shaders/frag.spv")};

        VkShaderModule vert_shader_module{
            create_shader_module(device_, vert_shader)};
        VkShaderModule frag_shader_module{
            create_shader_module(device_, frag_shader)};

        VkPipelineShaderStageCreateInfo vert_shader_stage_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader_module,
            .pName = "main"};

        VkPipelineShaderStageCreateInfo frag_shader_stage_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader_module,
            .pName = "main"};

        std::array shader_stages{vert_shader_stage_info,
            frag_shader_stage_info};

        VkPipelineVertexInputStateCreateInfo vertex_input_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr};

        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE};

        std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR};

        VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data()};

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f};

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE};

        VkPipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

        VkPipelineLayoutCreateInfo pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr};

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create pipeline layout"};
        }

        vkDestroyShaderModule(device_, frag_shader_module, nullptr);
        vkDestroyShaderModule(device_, vert_shader_module, nullptr);
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
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        std::ranges::for_each(swap_chain_image_views_,
            [this](VkImageView view)
            { vkDestroyImageView(device_, view, nullptr); });
        vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        window_.reset();
        glfwTerminate();
    }

private:
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_{nullptr,
        glfwDestroyWindow};
    VkInstance instance_;
    VkSurfaceKHR surface_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue graphics_queue_;
    VkQueue present_queue_;
    VkSwapchainKHR swap_chain_;
    std::vector<VkImage> swap_chain_images_;
    VkFormat swap_chain_image_format_;
    VkExtent2D swap_chain_extent_;
    std::vector<VkImageView> swap_chain_image_views_;
    VkRenderPass render_pass_;
    VkPipelineLayout pipeline_layout_;
};

int main()
{
    std::cout << "Working directory: " << std::filesystem::current_path()
              << '\n';

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
