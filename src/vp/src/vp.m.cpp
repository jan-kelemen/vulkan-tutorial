#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr std::string_view vertex_shader{"vert.spv"};
    constexpr std::string_view fragment_shader{"frag.spv"};
    constexpr std::string_view model_path{"viking_room.obj"};
    constexpr std::string_view texture_path{"viking_room.png"};

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
    constexpr std::array<char const*, 1> const device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    constexpr std::array<char const*, 1> const validation_layers{
        "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    constexpr bool enable_validation_layers{false};
#else
    constexpr bool enable_validation_layers{true};
#endif

    constexpr size_t max_frames_in_flight{2};

    void enumerate_extensions()
    {
        uint32_t count; // NOLINT
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

    [[nodiscard]] bool check_validation_layer_support()
    {
        uint32_t count; // NOLINT
        vkEnumerateInstanceLayerProperties(&count, nullptr);

        std::vector<VkLayerProperties> available_layers{count};
        vkEnumerateInstanceLayerProperties(&count, available_layers.data());

        for (auto const layer_name : validation_layers)
        {
            if (!std::ranges::any_of(available_layers,
                    [layer_name](VkLayerProperties const& layer)
                    { return strcmp(layer_name, layer.layerName) == 0; }))
            {
                return false;
            }
        }

        return true;
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT type,
        VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
        [[maybe_unused]] void* user_data)
    {
        std::cerr << "validation layer: " << callback_data->pMessage << '\n';
        return VK_FALSE;
    }

    VkResult create_debug_utils_messenger_ext(VkInstance instance,
        VkDebugUtilsMessengerCreateInfoEXT const* pCreateInfo,
        VkAllocationCallbacks const* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        // NOLINTNEXTLINE
        auto const func{reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"))};

        if (func != nullptr)
        {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        }

        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    void destroy_debug_utils_messenger_ext(VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        VkAllocationCallbacks const* pAllocator)
    {
        // NOLINTNEXTLINE
        auto const func{reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance,
                "vkDestroyDebugUtilsMessengerEXT"))};

        if (func != nullptr)
        {
            func(instance, debugMessenger, pAllocator);
        }
    }

    void populate_debug_messanger_create_info(
        VkDebugUtilsMessengerCreateInfoEXT& info)
    {
        info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = debug_callback;
    }

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

            VkBool32 present_support; // NOLINT
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

        uint32_t format_count; // NOLINT
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

        uint32_t present_count; // NOLINT
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

        int width; // NOLINT
        int height; // NOLINT
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
        uint32_t count; // NOLINT
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

        VkPhysicalDeviceFeatures supported_features;
        vkGetPhysicalDeviceFeatures(device, &supported_features);

        return indices.graphics_family && indices.present_family &&
            swap_chain_adequate && supported_features.samplerAnisotropy;
    }

    [[nodiscard]] VkShaderModule create_shader_module(VkDevice device,
        std::span<char const> code)
    {
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<uint32_t const*>(code.data());

        VkShaderModule module; // NOLINT
        if (vkCreateShaderModule(device, &create_info, nullptr, &module) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create shader module"};
        }

        return module;
    }

    [[nodiscard]] uint32_t find_memory_type(VkPhysicalDevice physical_device,
        uint32_t type_filter,
        VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device,
            &memory_properties);
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
        {
            if ((type_filter & (1 << i)) &&
                (memory_properties.memoryTypes[i].propertyFlags & properties) ==
                    properties)
            {
                return i;
            }
        }

        throw std::runtime_error{"failed to find suitable memory type!"};
    }

    VkCommandBuffer begin_single_time_commands(VkDevice device,
        VkCommandPool command_pool)
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer;
        if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to allocate command buffer!"};
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(command_buffer, &begin_info);

        return command_buffer;
    }

    void end_single_time_commands(VkDevice device,
        VkQueue graphics_queue,
        VkCommandBuffer command_buffer,
        VkCommandPool command_pool)
    {
        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command_buffer;

        vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue);

        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    }

    void create_buffer(VkPhysicalDevice physical_device,
        VkDevice device,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& buffer_memory)
    {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create buffer!"};
        }

        VkMemoryRequirements memory_requirements;
        vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = memory_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(physical_device,
            memory_requirements.memoryTypeBits,
            properties);

        if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to allocate buffer memory!"};
        }

        vkBindBufferMemory(device, buffer, buffer_memory, 0);
    }

    void copy_buffer(VkDevice device,
        VkCommandPool command_pool,
        VkQueue graphics_queue,
        VkBuffer source,
        VkBuffer target,
        VkDeviceSize size)
    {
        VkCommandBuffer command_buffer{
            begin_single_time_commands(device, command_pool)};

        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = size;
        vkCmdCopyBuffer(command_buffer, source, target, 1, &region);

        end_single_time_commands(device,
            graphics_queue,
            command_buffer,
            command_pool);
    }

    void create_image(VkPhysicalDevice physical_device,
        VkDevice device,
        uint32_t width,
        uint32_t height,
        uint32_t mip_levels,
        VkSampleCountFlagBits samples,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& image_memory)
    {
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = static_cast<uint32_t>(width);
        image_info.extent.height = static_cast<uint32_t>(height);
        image_info.extent.depth = 1;
        image_info.mipLevels = mip_levels;
        image_info.arrayLayers = 1;
        image_info.format = format;
        image_info.tiling = tiling;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = usage;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.samples = samples;
        image_info.flags = 0;

        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create image!"};
        }

        VkMemoryRequirements memory_requirements;
        vkGetImageMemoryRequirements(device, image, &memory_requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = memory_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(physical_device,
            memory_requirements.memoryTypeBits,
            properties);
        if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to allocate image memory!"};
        }
        vkBindImageMemory(device, image, image_memory, 0);
    }

    VkImageView create_image_view(VkDevice device,
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspect_flags,
        uint32_t mip_levels)
    {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = aspect_flags;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = mip_levels;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &view_info, nullptr, &imageView) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create texture image view!"};
        }

        return imageView;
    }

    void copy_buffer_to_image(VkDevice device,
        VkQueue graphics_queue,
        VkCommandPool command_pool,
        VkBuffer buffer,
        VkImage image,
        uint32_t width,
        uint32_t height)
    {
        VkCommandBuffer command_buffer{
            begin_single_time_commands(device, command_pool)};

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        vkCmdCopyBufferToImage(command_buffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);

        end_single_time_commands(device,
            graphics_queue,
            command_buffer,
            command_pool);
    }

    [[nodiscard]] VkFormat find_supported_format(
        VkPhysicalDevice physical_device,
        std::span<VkFormat const> candidates,
        VkImageTiling tiling,
        VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physical_device,
                format,
                &props);

            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                (props.optimalTilingFeatures & features) == features)
            {
                return format;
            }
        }

        throw std::runtime_error{"failed to find supported format!"};
    }

    [[nodiscard]] VkFormat find_depth_format(VkPhysicalDevice physical_device)
    {
        constexpr std::array candidates{VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT};
        return find_supported_format(physical_device,
            candidates,
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    [[nodiscard]] bool has_stencil_component(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void transition_image_layout(VkDevice device,
        VkQueue graphics_queue,
        VkCommandPool command_pool,
        VkImage image,
        VkFormat format,
        VkImageLayout old_layout,
        VkImageLayout new_layout,
        uint32_t mip_levels)
    {
        VkCommandBuffer command_buffer{
            begin_single_time_commands(device, command_pool)};

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mip_levels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (has_stencil_component(format))
            {
                barrier.subresourceRange.aspectMask |=
                    VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else
        {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        VkPipelineStageFlags source_stage;
        VkPipelineStageFlags destination_stage;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
            new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else
        {
            throw std::invalid_argument{"unsupported layout transition!"};
        }

        vkCmdPipelineBarrier(command_buffer,
            source_stage,
            destination_stage,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        end_single_time_commands(device,
            graphics_queue,
            command_buffer,
            command_pool);
    }

    void generate_mipmaps(VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue graphics_queue,
        VkCommandPool command_pool,
        VkImage image,
        VkFormat format,
        int32_t width,
        int32_t height,
        uint32_t mip_levels)
    {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physical_device,
            format,
            &properties);
        if (!(properties.optimalTilingFeatures &
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        {
            throw std::runtime_error{
                "texture image format does not support linear blitting!"};
        }

        VkCommandBuffer command_buffer{
            begin_single_time_commands(device, command_pool)};

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mip_width{width};
        int32_t mip_height{height};
        for (uint32_t i{1}; i != mip_levels; ++i)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &barrier);

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mip_width, mip_height, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1,
                mip_height > 1 ? mip_height / 2 : 1,
                1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(command_buffer,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &barrier);

            if (mip_width > 1)
            {
                mip_width /= 2;
            }
            if (mip_height > 1)
            {
                mip_height /= 2;
            }
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        end_single_time_commands(device,
            graphics_queue,
            command_buffer,
            command_pool);
    }

    VkSampleCountFlagBits max_usable_sample_count(
        VkPhysicalDevice physical_device)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical_device, &properties);

        VkSampleCountFlags const counts =
            properties.limits.framebufferColorSampleCounts &
            properties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT)
        {
            return VK_SAMPLE_COUNT_64_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_32_BIT)
        {
            return VK_SAMPLE_COUNT_32_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_16_BIT)
        {
            return VK_SAMPLE_COUNT_16_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_8_BIT)
        {
            return VK_SAMPLE_COUNT_8_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_4_BIT)
        {
            return VK_SAMPLE_COUNT_4_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_2_BIT)
        {
            return VK_SAMPLE_COUNT_2_BIT;
        }

        return VK_SAMPLE_COUNT_1_BIT;
    }
} // namespace

namespace
{
    struct vertex
    {
        glm::vec3 pos;
        glm::vec3 color;
        glm::vec2 tex_coord;

        static constexpr VkVertexInputBindingDescription binding_description()
        {
            VkVertexInputBindingDescription desc{};
            desc.binding = 0;
            desc.stride = sizeof(vertex);
            desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            return desc;
        }

        static constexpr std::array<VkVertexInputAttributeDescription, 3>
        attribute_description()
        {
            std::array<VkVertexInputAttributeDescription, 3> desc{};

            desc[0].binding = 0;
            desc[0].location = 0;
            desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            desc[0].offset = offsetof(vertex, pos);

            desc[1].binding = 0;
            desc[1].location = 1;
            desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            desc[1].offset = offsetof(vertex, color);

            desc[2].binding = 0;
            desc[2].location = 2;
            desc[2].format = VK_FORMAT_R32G32_SFLOAT;
            desc[2].offset = offsetof(vertex, tex_coord);

            return desc;
        }

        friend bool operator==(vertex const&, vertex const&) = default;
    };

    struct uniform_buffer_object
    {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };
} // namespace

template<>
struct std::hash<vertex>
{
    size_t operator()(vertex const& vert) const
    {
        return ((hash<glm::vec3>()(vert.pos) ^
                    (hash<glm::vec3>()(vert.color) << 1)) >>
                   1) ^
            (hash<glm::vec2>()(vert.tex_coord) << 1);
    }
};

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
        window_.reset(
            glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr));
        glfwSetWindowUserPointer(window_.get(), this);
        glfwSetFramebufferSizeCallback(window_.get(),
            framebuffer_resize_callback);
    }

    static void framebuffer_resize_callback(GLFWwindow* window,
        [[maybe_unused]] int width,
        [[maybe_unused]] int height)
    {
        auto app{reinterpret_cast<hello_triangle_application*>(
            glfwGetWindowUserPointer(window))};
        app->framebuffer_resized = true;
    }

    void init_vulkan()
    {
        bool setup_debug{enable_validation_layers};
        create_instance(setup_debug);
        if (setup_debug)
        {
            setup_debug_messenger();
        }
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swap_chain();
        create_image_views();
        create_render_pass();
        create_descriptor_set_layout();
        create_graphics_pipeline();
        create_command_pool();
        create_color_resources();
        create_depth_resources();
        create_framebuffers();
        create_texture_image();
        create_texture_image_view();
        create_texture_sampler();
        load_model();
        create_vertex_buffer();
        create_index_buffer();
        create_uniform_buffers();
        create_descriptor_pool();
        create_descriptor_sets();
        create_command_buffers();
        create_sync_objects();
    }

    void create_instance(bool& setup_debug)
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

        uint32_t glfw_extension_count; // NOLINT
        char const** const glfw_extensions{
            glfwGetRequiredInstanceExtensions(&glfw_extension_count)};

        std::vector<char const*> required_extensions{glfw_extensions,
            glfw_extensions + glfw_extension_count};

        if (enable_validation_layers)
        {
            if (check_validation_layer_support())
            {
                create_info.enabledLayerCount =
                    static_cast<uint32_t>(validation_layers.size());
                create_info.ppEnabledLayerNames = validation_layers.data();

                required_extensions.push_back(
                    VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

                VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
                populate_debug_messanger_create_info(debug_create_info);
                create_info.pNext = &debug_create_info;
            }
            else
            {
                std::cerr << "Validation layers requested but not available!\n";
                setup_debug = false;
            }
        }
        else
        {
            create_info.enabledLayerCount = 0;
        }

        create_info.enabledExtensionCount =
            static_cast<uint32_t>(required_extensions.size());
        create_info.ppEnabledExtensionNames = required_extensions.data();

        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void setup_debug_messenger()
    {
        VkDebugUtilsMessengerCreateInfoEXT create_info;
        populate_debug_messanger_create_info(create_info);

        VkDebugUtilsMessengerEXT messenger_; // NOLINT
        if (create_debug_utils_messenger_ext(instance_,
                &create_info,
                nullptr,
                &messenger_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to setup debug messenger!"};
        }

        debug_messenger_ = messenger_;
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
        msaa_samples_ = max_usable_sample_count(physical_device_);
    }

    void create_logical_device()
    {
        queue_family_indices queue_indices =
            find_queue_families(physical_device_, surface_);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> unique_families = {
            queue_indices.graphics_family.value(),
            queue_indices.present_family.value()};
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
        device_features.samplerAnisotropy = VK_TRUE;
        device_features.sampleRateShading = VK_TRUE;

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
            queue_indices.graphics_family.value(),
            0,
            &graphics_queue_);

        vkGetDeviceQueue(device_,
            queue_indices.present_family.value(),
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

        queue_family_indices const queue_indices{
            find_queue_families(physical_device_, surface_)};
        std::array queue_family_indices{queue_indices.present_family.value(),
            queue_indices.present_family.value()};

        if (queue_indices.graphics_family != queue_indices.present_family)
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

    void recreate_swap_chain()
    {
        int width{};
        int height{};
        glfwGetFramebufferSize(window_.get(), &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window_.get(), &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device_);

        cleanup_swap_chain();

        create_swap_chain();
        create_image_views();
        create_depth_resources();
        create_framebuffers();
    }

    void create_image_views()
    {
        swap_chain_image_views_.resize(swap_chain_images_.size());
        for (size_t i{}; i != swap_chain_images_.size(); ++i)
        {
            swap_chain_image_views_[i] = create_image_view(device_,
                swap_chain_images_[i],
                swap_chain_image_format_,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1);
        }
    }

    void create_render_pass()
    {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = swap_chain_image_format_;
        color_attachment.samples = msaa_samples_;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depth_attachment{};
        depth_attachment.format = find_depth_format(physical_device_);
        depth_attachment.samples = msaa_samples_;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_attachment_ref{};
        depth_attachment_ref.attachment = 1;
        depth_attachment_ref.layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription color_attachment_resolve{};
        color_attachment_resolve.format = swap_chain_image_format_;
        color_attachment_resolve.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment_resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment_resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment_resolve.stencilLoadOp =
            VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment_resolve.stencilStoreOp =
            VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment_resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment_resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_resolve_ref{};
        color_attachment_resolve_ref.attachment = 2;
        color_attachment_resolve_ref.layout =
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;
        subpass.pDepthStencilAttachment = &depth_attachment_ref;
        subpass.pResolveAttachments = &color_attachment_resolve_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array attachments{color_attachment,
            depth_attachment,
            color_attachment_resolve};
        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount =
            static_cast<uint32_t>(attachments.size());
        render_pass_info.pAttachments = attachments.data();
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

    void create_descriptor_set_layout()
    {
        VkDescriptorSetLayoutBinding ubo_layout_binding{};
        ubo_layout_binding.binding = 0;
        ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_layout_binding.descriptorCount = 1;
        ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        ubo_layout_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding sampler_layout_binding{};
        sampler_layout_binding.binding = 1;
        sampler_layout_binding.descriptorCount = 1;
        sampler_layout_binding.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sampler_layout_binding.pImmutableSamplers = nullptr;

        std::array bindings{ubo_layout_binding, sampler_layout_binding};
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device_,
                &layout_info,
                nullptr,
                &descriptor_set_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create descriptor set layout"};
        }
    }

    void create_graphics_pipeline()
    {
        auto const vert_shader{read_file(vertex_shader)};
        auto const frag_shader{read_file(fragment_shader)};

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

        auto const binding_description{vertex::binding_description()};
        auto const attribute_description{vertex::attribute_description()};
        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &binding_description;
        vertex_input_info.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attribute_description.size());
        vertex_input_info.pVertexAttributeDescriptions =
            attribute_description.data();

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
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = msaa_samples_;
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.minSampleShading = .2f;
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

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depth_stencil.depthBoundsTestEnable = VK_FALSE;
        depth_stencil.minDepthBounds = 0.0f;
        depth_stencil.maxDepthBounds = 1.0f;
        depth_stencil.stencilTestEnable = VK_FALSE;
        depth_stencil.front = {};
        depth_stencil.back = {};

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
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
        pipeline_info.pDepthStencilState = &depth_stencil;
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
            std::array attachments{color_image_view_,
                depth_image_view_,
                swap_chain_image_views_[i]};

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
        queue_family_indices queue_indices{
            find_queue_families(physical_device_, surface_)};

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        pool_info.queueFamilyIndex = *queue_indices.graphics_family;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
            VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create command pool"};
        }
    }

    void create_color_resources()
    {
        VkFormat const color_format{swap_chain_image_format_};

        create_image(physical_device_,
            device_,
            swap_chain_extent_.width,
            swap_chain_extent_.height,
            1,
            msaa_samples_,
            color_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            color_image_,
            color_image_memory_);
        color_image_view_ = create_image_view(device_,
            color_image_,
            color_format,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1);
    }

    void create_depth_resources()
    {
        VkFormat const depth_format{find_depth_format(physical_device_)};

        create_image(physical_device_,
            device_,
            swap_chain_extent_.width,
            swap_chain_extent_.height,
            1,
            msaa_samples_,
            depth_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depth_image_,
            depth_image_memory_);

        depth_image_view_ = create_image_view(device_,
            depth_image_,
            depth_format,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            1);

        transition_image_layout(device_,
            graphics_queue_,
            command_pool_,
            depth_image_,
            depth_format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1);
    }

    void create_texture_image()
    {
        int width;
        int height;
        int channels;
        stbi_uc* pixels{stbi_load(texture_path.data(),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha)};

        auto const image_size{static_cast<VkDeviceSize>(width * height * 4)};
        mip_levels_ = static_cast<uint32_t>(
                          std::floor(std::log2(std::max(width, height)))) +
            1;

        if (!pixels)
        {
            throw std::runtime_error{"failed to load texture image!"};
        }

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        create_buffer(physical_device_,
            device_,
            image_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        vkMapMemory(device_, staging_buffer_memory, 0, image_size, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(image_size));
        vkUnmapMemory(device_, staging_buffer_memory);
        stbi_image_free(pixels);

        create_image(physical_device_,
            device_,
            width,
            height,
            mip_levels_,
            VK_SAMPLE_COUNT_1_BIT,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            texture_image_,
            texture_image_memory_);

        transition_image_layout(device_,
            graphics_queue_,
            command_pool_,
            texture_image_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mip_levels_);
        copy_buffer_to_image(device_,
            graphics_queue_,
            command_pool_,
            staging_buffer,
            texture_image_,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height));
        generate_mipmaps(physical_device_,
            device_,
            graphics_queue_,
            command_pool_,
            texture_image_,
            VK_FORMAT_R8G8B8A8_SRGB,
            width,
            height,
            mip_levels_);

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void create_texture_image_view()
    {
        texture_image_view_ = create_image_view(device_,
            texture_image_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_ASPECT_COLOR_BIT,
            mip_levels_);
    }

    void create_texture_sampler()
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device_, &properties);

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.anisotropyEnable = VK_TRUE;
        sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = VK_FALSE;
        sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.mipLodBias = 0.0f;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = static_cast<float>(mip_levels_);

        if (vkCreateSampler(device_,
                &sampler_info,
                nullptr,
                &texture_sampler_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create_texture sampler!"};
        }
    }

    void load_model()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warning;
        std::string error;

        if (!tinyobj::LoadObj(&attrib,
                &shapes,
                &materials,
                &warning,
                &error,
                model_path.data()))
        {
            throw std::runtime_error{warning + error};
        }

        std::unordered_map<vertex, uint32_t> unique_vertices;

        for (auto const& shape : shapes)
        {
            for (auto const& index : shape.mesh.indices)
            {
                vertex vert;

                vert.pos = {attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]};

                vert.tex_coord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

                vert.color = {1.0f, 1.0f, 1.0f};

                auto const& [it, inserted] = unique_vertices.try_emplace(vert,
                    static_cast<uint32_t>(vertices_.size()));
                if (inserted)
                {
                    vertices_.push_back(vert);
                }

                indices_.push_back(it->second);
            }
        }
    }

    void create_vertex_buffer()
    {
        VkDeviceSize const buffer_size{sizeof(vertices_[0]) * vertices_.size()};

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        create_buffer(physical_device_,
            device_,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
        memcpy(data, vertices_.data(), static_cast<size_t>(buffer_size));
        vkUnmapMemory(device_, staging_buffer_memory);

        create_buffer(physical_device_,
            device_,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertex_buffer_,
            vertex_buffer_memory_);

        copy_buffer(device_,
            command_pool_,
            graphics_queue_,
            staging_buffer,
            vertex_buffer_,
            buffer_size);

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void create_index_buffer()
    {
        VkDeviceSize buffer_size{sizeof(indices_[0]) * indices_.size()};

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        create_buffer(physical_device_,
            device_,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        vkMapMemory(device_, staging_buffer_memory, 0, buffer_size, 0, &data);
        memcpy(data, indices_.data(), static_cast<size_t>(buffer_size));
        vkUnmapMemory(device_, staging_buffer_memory);

        create_buffer(physical_device_,
            device_,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            index_buffer_,
            index_buffer_memory_);

        copy_buffer(device_,
            command_pool_,
            graphics_queue_,
            staging_buffer,
            index_buffer_,
            buffer_size);

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_buffer_memory, nullptr);
    }

    void create_uniform_buffers()
    {
        VkDeviceSize const buffer_size{sizeof(uniform_buffer_object)};

        uniform_buffers_.resize(max_frames_in_flight);
        uniform_buffers_memory_.resize(max_frames_in_flight);
        uniform_buffers_mapped_.resize(max_frames_in_flight);

        for (size_t i{}; i < max_frames_in_flight; ++i)
        {
            create_buffer(physical_device_,
                device_,
                buffer_size,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniform_buffers_[i],
                uniform_buffers_memory_[i]);

            if (vkMapMemory(device_,
                    uniform_buffers_memory_[i],
                    0,
                    buffer_size,
                    0,
                    &uniform_buffers_mapped_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error{
                    "failed to map uniform buffer memory!"};
            }
        }
    }

    void create_descriptor_pool()
    {
        VkDescriptorPoolSize uniform_buffer_pool{};
        uniform_buffer_pool.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniform_buffer_pool.descriptorCount =
            static_cast<uint32_t>(max_frames_in_flight);

        VkDescriptorPoolSize sampler_pool{};
        sampler_pool.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_pool.descriptorCount =
            static_cast<uint32_t>(max_frames_in_flight);

        std::array pool_sizes{uniform_buffer_pool, sampler_pool};

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = static_cast<uint32_t>(max_frames_in_flight);

        if (vkCreateDescriptorPool(device_,
                &pool_info,
                nullptr,
                &descriptor_pool_) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to create descriptor pool!"};
        }
    }

    void create_descriptor_sets()
    {
        std::vector<VkDescriptorSetLayout> layouts(max_frames_in_flight,
            descriptor_set_layout_);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptor_pool_;
        allocInfo.descriptorSetCount =
            static_cast<uint32_t>(max_frames_in_flight);
        allocInfo.pSetLayouts = layouts.data();

        descriptor_sets_.resize(max_frames_in_flight);
        if (vkAllocateDescriptorSets(device_,
                &allocInfo,
                descriptor_sets_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < max_frames_in_flight; i++)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniform_buffers_[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(uniform_buffer_object);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = texture_image_view_;
            imageInfo.sampler = texture_sampler_;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptor_sets_[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType =
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptor_sets_[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device_,
                static_cast<uint32_t>(descriptorWrites.size()),
                descriptorWrites.data(),
                0,
                nullptr);
        }
    }

    void create_command_buffers()
    {
        command_buffers_.resize(max_frames_in_flight);

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount =
            static_cast<uint32_t>(command_buffers_.size());

        if (vkAllocateCommandBuffers(device_,
                &alloc_info,
                command_buffers_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to allocate command buffer!"};
        }
    }

    void create_sync_objects()
    {
        image_available_semaphores_.resize(max_frames_in_flight);
        render_finished_semaphores_.resize(max_frames_in_flight);
        in_flight_fences_.resize(max_frames_in_flight);

        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i{}; i != max_frames_in_flight; ++i)
        {
            std::array result{vkCreateSemaphore(device_,
                                  &semaphore_info,
                                  nullptr,
                                  &image_available_semaphores_[i]),
                vkCreateSemaphore(device_,
                    &semaphore_info,
                    nullptr,
                    &render_finished_semaphores_[i]),
                vkCreateFence(device_,
                    &fence_info,
                    nullptr,
                    &in_flight_fences_[i])};

            if (static_cast<size_t>(std::ranges::count(result, VK_SUCCESS)) !=
                result.size())
            {
                throw std::runtime_error{"failed to create sync objects"};
            }
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

        std::array<VkClearValue, 2> clear_values{};
        clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clear_values[1].color = {1.0f, 0};

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = render_pass_;
        render_pass_info.framebuffer = swap_chain_framebuffers_[image_index];
        render_pass_info.renderArea = {{0, 0}, swap_chain_extent_};
        render_pass_info.clearValueCount =
            static_cast<uint32_t>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();

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

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swap_chain_extent_;
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        VkBuffer vertex_buffers[] = {vertex_buffer_};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

        vkCmdBindIndexBuffer(command_buffer,
            index_buffer_,
            0,
            VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &descriptor_sets_[current_frame_],
            0,
            nullptr);
        vkCmdDrawIndexed(command_buffer,
            static_cast<uint32_t>(indices_.size()),
            1,
            0,
            0,
            0);

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

        vkWaitForFences(device_,
            1,
            &in_flight_fences_[current_frame_],
            VK_TRUE,
            timeout);

        uint32_t image_index; // NOLINT
        VkResult result{vkAcquireNextImageKHR(device_,
            swap_chain_,
            timeout,
            image_available_semaphores_[current_frame_],
            VK_NULL_HANDLE,
            &image_index)};

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreate_swap_chain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error{"failed to acquire swap chain image"};
        }

        vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

        vkResetCommandBuffer(command_buffers_[current_frame_], 0);
        record_command_buffer(command_buffers_[current_frame_], image_index);

        update_uniform_buffer(current_frame_);

        std::array wait_semaphores{image_available_semaphores_[current_frame_]};
        std::array<VkPipelineStageFlags, 1> wait_stages{
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        std::array signal_semaphores{
            render_finished_semaphores_[current_frame_]};
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount =
            static_cast<uint32_t>(wait_semaphores.size());
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers_[current_frame_];
        submit_info.signalSemaphoreCount =
            static_cast<uint32_t>(signal_semaphores.size());
        submit_info.pSignalSemaphores = signal_semaphores.data();
        if (vkQueueSubmit(graphics_queue_,
                1,
                &submit_info,
                in_flight_fences_[current_frame_]) != VK_SUCCESS)
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
        result = vkQueuePresentKHR(present_queue_, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
            framebuffer_resized)
        {
            framebuffer_resized = false;
            recreate_swap_chain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error{"failed to present swap chain image!"};
        }

        current_frame_ = (current_frame_ + 1) % max_frames_in_flight;
    }

    void update_uniform_buffer(uint32_t current_image)
    {
        using namespace std::chrono;

        static auto const start_time{high_resolution_clock::now()};

        auto const current_time{high_resolution_clock::now()};
        float const time =
            duration<float, seconds::period>(current_time - start_time).count();

        uniform_buffer_object ubo{};
        ubo.model = glm::rotate(glm::mat4{1.0f},
            time * glm::radians(90.0f),
            glm::vec3{0.0f, 0.0f, 1.0f});

        ubo.view = glm::lookAt(glm::vec3{2.0f, 2.0f, 2.0f},
            glm::vec3{0.0f, 0.0f, 0.0f},
            glm::vec3{0.0f, 0.0f, 1.0f});

        ubo.proj = glm::perspective(glm::radians(45.0f),
            swap_chain_extent_.width /
                static_cast<float>(swap_chain_extent_.height),
            0.1f,
            10.0f);

        ubo.proj[1][1] *= -1;

        memcpy(uniform_buffers_mapped_[current_image], &ubo, sizeof(ubo));
    }

    void cleanup()
    {
        cleanup_swap_chain();

        vkDestroySampler(device_, texture_sampler_, nullptr);

        vkDestroyImageView(device_, texture_image_view_, nullptr);
        vkDestroyImage(device_, texture_image_, nullptr);
        vkFreeMemory(device_, texture_image_memory_, nullptr);

        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);

        for (size_t i{}; i != max_frames_in_flight; ++i)
        {
            vkDestroyBuffer(device_, uniform_buffers_[i], nullptr);
            vkFreeMemory(device_, uniform_buffers_memory_[i], nullptr);
        }

        vkDestroyBuffer(device_, index_buffer_, nullptr);
        vkFreeMemory(device_, index_buffer_memory_, nullptr);

        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);

        vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);

        vkDestroyRenderPass(device_, render_pass_, nullptr);

        for (size_t i{}; i != max_frames_in_flight; ++i)
        {
            vkDestroySemaphore(device_,
                image_available_semaphores_[i],
                nullptr);
            vkDestroySemaphore(device_,
                render_finished_semaphores_[i],
                nullptr);
            vkDestroyFence(device_, in_flight_fences_[i], nullptr);
        }

        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (debug_messenger_)
        {
            destroy_debug_utils_messenger_ext(instance_,
                *debug_messenger_,
                nullptr);
        }
        vkDestroyInstance(instance_, nullptr);
        window_.reset();
        glfwTerminate();
    }

    void cleanup_swap_chain()
    {
        vkDestroyImageView(device_, color_image_view_, nullptr);
        vkDestroyImage(device_, color_image_, nullptr);
        vkFreeMemory(device_, color_image_memory_, nullptr);

        vkDestroyImageView(device_, depth_image_view_, nullptr);
        vkDestroyImage(device_, depth_image_, nullptr);
        vkFreeMemory(device_, depth_image_memory_, nullptr);

        for (size_t i{}; i != swap_chain_framebuffers_.size(); ++i)
        {
            vkDestroyFramebuffer(device_, swap_chain_framebuffers_[i], nullptr);
        }

        for (size_t i{}; i != swap_chain_image_views_.size(); ++i)
        {
            vkDestroyImageView(device_, swap_chain_image_views_[i], nullptr);
        }

        vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
    }

private:
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_{nullptr,
        glfwDestroyWindow};
    VkInstance instance_;
    std::optional<VkDebugUtilsMessengerEXT> debug_messenger_;
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
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline graphics_pipeline_;
    std::vector<VkFramebuffer> swap_chain_framebuffers_;
    VkCommandPool command_pool_;
    uint32_t mip_levels_;
    VkImage texture_image_;
    VkDeviceMemory texture_image_memory_;
    VkImageView texture_image_view_;
    VkImage color_image_;
    VkDeviceMemory color_image_memory_;
    VkImageView color_image_view_;
    VkImage depth_image_;
    VkDeviceMemory depth_image_memory_;
    VkImageView depth_image_view_;
    VkSampler texture_sampler_;
    std::vector<vertex> vertices_;
    std::vector<uint32_t> indices_;
    VkBuffer vertex_buffer_;
    VkDeviceMemory vertex_buffer_memory_;
    VkBuffer index_buffer_;
    VkDeviceMemory index_buffer_memory_;
    std::vector<VkBuffer> uniform_buffers_;
    std::vector<VkDeviceMemory> uniform_buffers_memory_;
    std::vector<void*> uniform_buffers_mapped_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    VkDescriptorPool descriptor_pool_;
    std::vector<VkDescriptorSet> descriptor_sets_;
    VkSampleCountFlagBits msaa_samples_{VK_SAMPLE_COUNT_1_BIT};

    uint32_t current_frame_{};
    bool framebuffer_resized{};
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
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
