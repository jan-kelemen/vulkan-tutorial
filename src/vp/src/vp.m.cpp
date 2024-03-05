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

        auto const eof{stream.tellg()};

        std::vector<char> buffer(static_cast<size_t>(eof));
        stream.seekg(0);

        stream.read(buffer.data(), eof);

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
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<uint32_t const*>(code.data());

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
        create_framebuffers();
        create_command_pool();
        create_command_buffer();
        create_sync_objects();
    }

    void create_instance()
    {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Hello Triangle";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

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
            VkDeviceQueueCreateInfo queue_create_info{};
            queue_create_info.sType =
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_create_info.queueFamilyIndex = family;
            queue_create_info.queueCount = 1;
            queue_create_info.pQueuePriorities = &priority;

            queue_create_infos.push_back(queue_create_info);
        }

        VkPhysicalDeviceFeatures device_features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount =
            static_cast<uint32_t>(queue_create_infos.size());
        create_info.pQueueCreateInfos = queue_create_infos.data();
        create_info.enabledLayerCount = 0;
        create_info.enabledExtensionCount =
            static_cast<uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.data();
        create_info.pEnabledFeatures = &device_features;

        if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) !=
            VK_SUCCESS)
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

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create_info.preTransform = support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = VK_NULL_HANDLE;

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
            VkImageViewCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = image;
            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = swap_chain_image_format_;
            create_info.components = components;
            create_info.subresourceRange = subresource_range;

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
        VkAttachmentDescription color_attachment{};
        color_attachment.format = swap_chain_image_format_;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        if (vkCreateRenderPass(device_,
                &render_pass_info,
                nullptr,
                &render_pass_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create render pass"};
        }
    }

    void create_graphics_pipeline()
    {
        auto const vert_shader{read_file("vert.spv")};
        auto const frag_shader{read_file("frag.spv")};

        VkShaderModule vert_shader_module{
            create_shader_module(device_, vert_shader)};
        VkShaderModule frag_shader_module{
            create_shader_module(device_, frag_shader)};

        VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
        vert_shader_stage_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_shader_stage_info.module = vert_shader_module;
        vert_shader_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
        frag_shader_stage_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_shader_stage_info.module = frag_shader_module;
        frag_shader_stage_info.pName = "main";

        std::array shader_stages{vert_shader_stage_info,
            frag_shader_stage_info};

        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 0;
        vertex_input_info.pVertexBindingDescriptions = nullptr;
        vertex_input_info.vertexAttributeDescriptionCount = 0;
        vertex_input_info.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR};

        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        dynamic_state.dynamicStateCount =
            static_cast<uint32_t>(dynamic_states.size()),
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment.blendEnable = VK_FALSE,
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        std::ranges::fill(color_blending.blendConstants, 0.0f);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pSetLayouts = nullptr;
        pipeline_layout_info.pushConstantRangeCount = 0;
        pipeline_layout_info.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device_,
                &pipeline_layout_info,
                nullptr,
                &pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create pipeline layout"};
        }

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages.data();
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pDepthStencilState = nullptr;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_state;
        pipeline_info.layout = pipeline_layout_;
        pipeline_info.renderPass = render_pass_;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device_,
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &graphics_pipeline_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create graphics pipeline!"};
        }

        vkDestroyShaderModule(device_, frag_shader_module, nullptr);
        vkDestroyShaderModule(device_, vert_shader_module, nullptr);
    }

    void create_framebuffers()
    {
        swap_chain_framebuffers_.resize(swap_chain_image_views_.size());

        for (size_t i{}; i != swap_chain_image_views_.size(); ++i)
        {
            std::array attachments{swap_chain_image_views_[i]};

            VkFramebufferCreateInfo frame_buffer_info{};
            frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            frame_buffer_info.renderPass = render_pass_;
            frame_buffer_info.attachmentCount =
                static_cast<uint32_t>(attachments.size());
            frame_buffer_info.pAttachments = attachments.data();
            frame_buffer_info.width = swap_chain_extent_.width;
            frame_buffer_info.height = swap_chain_extent_.height;
            frame_buffer_info.layers = 1;

            if (vkCreateFramebuffer(device_,
                    &frame_buffer_info,
                    nullptr,
                    &swap_chain_framebuffers_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error{"failed to create framebuffer!"};
            }
        }
    }

    void create_command_pool()
    {
        queue_family_indices indices{
            find_queue_families(physical_device_, surface_)};

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        pool_info.queueFamilyIndex = *indices.graphics_family;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create command pool"};
        }
    }

    void create_command_buffer()
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to allocate command buffer!"};
        }
    }

    void create_sync_objects()
    {
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        std::array result{vkCreateSemaphore(device_,
                              &semaphore_info,
                              nullptr,
                              &image_available_semaphore_),
            vkCreateSemaphore(device_,
                &semaphore_info,
                nullptr,
                &render_finished_semaphore_),
            vkCreateFence(device_, &fence_info, nullptr, &in_flight_fence_)};

        if (static_cast<size_t>(std::ranges::count(result, VK_SUCCESS)) !=
            result.size())
        {
            throw std::runtime_error{"failed to create sync objects"};
        }
    }

    void record_command_buffer(VkCommandBuffer command_buffer,
        uint32_t image_index)
    {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = 0;
        begin_info.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        {
            throw std::runtime_error{
                "failed to begin recording command buffer!"};
        }

        VkClearValue clearColor{{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = render_pass_;
        render_pass_info.framebuffer = swap_chain_framebuffers_[image_index];
        render_pass_info.renderArea = {{0, 0}, swap_chain_extent_};
        render_pass_info.clearValueCount = 1;
        render_pass_info.pClearValues = &clearColor;

        vkCmdBeginRenderPass(command_buffer,
            &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            graphics_pipeline_);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swap_chain_extent_.width);
        viewport.height = static_cast<float>(swap_chain_extent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        VkRect2D scissor{.offset = {0, 0}, .extent = swap_chain_extent_};
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(command_buffer);

        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to record command buffer!"};
        }
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window_.get()))
        {
            glfwPollEvents();
            draw_frame();
        }

        vkDeviceWaitIdle(device_);
    }

    void draw_frame()
    {
        constexpr auto timeout{std::numeric_limits<uint64_t>::max()};

        vkWaitForFences(device_, 1, &in_flight_fence_, VK_TRUE, timeout);
        vkResetFences(device_, 1, &in_flight_fence_);

        uint32_t image_index;
        vkAcquireNextImageKHR(device_,
            swap_chain_,
            timeout,
            image_available_semaphore_,
            VK_NULL_HANDLE,
            &image_index);

        vkResetCommandBuffer(command_buffer_, 0);
        record_command_buffer(command_buffer_, image_index);

        std::array wait_semaphores{image_available_semaphore_};
        std::array<VkPipelineStageFlags, 1> wait_stages{
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        std::array signal_semaphores{render_finished_semaphore_};
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount =
            static_cast<uint32_t>(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer_;
        submit_info.signalSemaphoreCount =
            static_cast<uint32_t>(signal_semaphores.size());
        submit_info.pSignalSemaphores = signal_semaphores.data();

        if (vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_fence_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to submit draw command buffer"};
        }

        std::array swap_chains{swap_chain_};
        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount =
            static_cast<uint32_t>(signal_semaphores.size());
        present_info.pWaitSemaphores = signal_semaphores.data();
        present_info.swapchainCount = static_cast<uint32_t>(swap_chains.size());
        present_info.pSwapchains = swap_chains.data();
        present_info.pImageIndices = &image_index;
        present_info.pResults = nullptr;
        vkQueuePresentKHR(present_queue_, &present_info);
    }

    void cleanup()
    {
        vkDestroySemaphore(device_, image_available_semaphore_, nullptr);
        vkDestroySemaphore(device_, render_finished_semaphore_, nullptr);
        vkDestroyFence(device_, in_flight_fence_, nullptr);
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        std::ranges::for_each(swap_chain_framebuffers_,
            [this](VkFramebuffer buffer)
            { vkDestroyFramebuffer(device_, buffer, nullptr); });
        vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
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
    VkPipeline graphics_pipeline_;
    std::vector<VkFramebuffer> swap_chain_framebuffers_;
    VkCommandPool command_pool_;
    VkCommandBuffer command_buffer_;
    VkSemaphore image_available_semaphore_;
    VkSemaphore render_finished_semaphore_;
    VkFence in_flight_fence_;
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
