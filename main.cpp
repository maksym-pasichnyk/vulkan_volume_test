#define VK_NO_PROTOTYPES

#include <map>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <SDL_video.h>
#include <SDL_events.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

vk::DispatchLoaderDynamic vk::defaultDispatchLoaderDynamic;

template<typename Fn>
struct lazy : Fn {
    explicit operator auto() && {
        return (*this)();
    }
};

template<typename Fn>
lazy(Fn&&) -> lazy<Fn>;

struct ShaderUniforms {
    alignas(16) glm::mat4x4 world_to_clip;
    alignas(16) glm::vec3   camera_position;
};

struct VulkanApplication {
    u32 max_frames_in_flight = 3;

    SDL_Window* window;
    vk::DynamicLoader loader;
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::DebugUtilsMessengerEXT messenger;
    vk::Device logical_device;
    vk::PhysicalDevice physical_device;
    vk::Queue graphics_queue;
    u32 graphics_queue_family_index;

    vk::Extent2D swapchain_extent;
    vk::SurfaceFormatKHR surface_format;

    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_image_views;

    std::vector<vk::Image> depth_images;
    std::vector<vk::ImageView> depth_image_views;
    std::vector<vk::DeviceMemory> depth_image_memories;

    std::vector<vk::Fence> in_flight_fences;
    std::vector<vk::Semaphore> image_available_semaphores;
    std::vector<vk::Semaphore> render_finished_semaphores;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers;

    vk::DescriptorPool descriptor_pool;

    std::map<std::string, vk::ShaderModule> shader_modules;

    u32 frame_index = 0;
    u32 image_index = 0;

    glm::vec3 camera_position;
    glm::vec3 camera_rotation;

    struct {
        vk::Buffer          vertex_buffer;
        vk::DeviceMemory    vertex_buffer_memory;

        vk::Buffer          index_buffer;
        vk::DeviceMemory    index_buffer_memory;
    } geometry;

    struct {
        vk::Image image;
        vk::ImageView view;
        vk::Sampler sampler;
        vk::DeviceMemory memory;
    } car_volume_texture;

    struct {
        vk::Pipeline graphics_pipeline;
        vk::PipelineLayout pipeline_layout;
        vk::DescriptorSet descriptor_set;
        vk::DescriptorSetLayout descriptor_set_layout;
    } unlit_normal_material;

    VulkanApplication() {
        window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI);

        camera_position = glm::vec3(0.0f, 1.0f, -5.0f);
        camera_rotation = glm::vec3(0.0f, 0.0f, 0.0f);

        initInstance();
        initLogicalDevice();
        initSwapchain();
        initSyncObjects();
        initCommandPool();
        initCommandBuffers();
        initDescriptorPool();
        initGraphicsPipelines();
        initCarVolumeTexture();
    }

    ~VulkanApplication() {
        logical_device.waitIdle();

        destroyCarVolumeTexture();
        destroyShaderModules();
        destroyGraphicsPipelines();
        destroyDescriptorPool();
        destroyCommandBuffers();
        destroyCommandPool();
        destroySyncObjects();
        destroySwapchain();
        destroyLogicalDevice();
        destroyInstance();

        SDL_DestroyWindow(window);
    }

    void initInstance() {
        vk::defaultDispatchLoaderDynamic.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

        vk::ApplicationInfo app_info = {};
        app_info.pApplicationName = "Vulkan";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "Vulkan";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char*> instance_layers = {};
        std::vector<const char*> instance_extensions = {};

        instance_layers.push_back("VK_LAYER_KHRONOS_validation");
        instance_layers.push_back("VK_LAYER_KHRONOS_synchronization2");

        instance_extensions.push_back("VK_KHR_surface");
        instance_extensions.push_back("VK_EXT_debug_utils");
        instance_extensions.push_back("VK_MVK_macos_surface");
        instance_extensions.push_back("VK_KHR_portability_enumeration");

        vk::InstanceCreateInfo instance_info = {};
        instance_info.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        instance_info.pApplicationInfo = &app_info;
        instance_info.enabledLayerCount = instance_layers.size();
        instance_info.ppEnabledLayerNames = instance_layers.data();
        instance_info.enabledExtensionCount = instance_extensions.size();
        instance_info.ppEnabledExtensionNames = instance_extensions.data();

        vk::resultCheck(vk::createInstance(&instance_info, nullptr, &instance), "Failed to create Vulkan instance");

        vk::defaultDispatchLoaderDynamic.init(instance);

        vk::DebugUtilsMessengerCreateInfoEXT messenger_info = {};
        messenger_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
        messenger_info.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
        messenger_info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user_data) -> VkBool32 {
            std::cout << data->pMessage << std::endl;
            return VK_FALSE;
        };

        vk::resultCheck(instance.createDebugUtilsMessengerEXT(&messenger_info, nullptr, &messenger), "Failed to create debug messenger");

        SDL_Vulkan_CreateSurface(window, instance, reinterpret_cast<VkSurfaceKHR*>(&surface));
    }

    void destroyInstance() {
        instance.destroySurfaceKHR(surface);
        instance.destroy(messenger);
        instance.destroy();
    }

    void initLogicalDevice() {
        vk::defaultDispatchLoaderDynamic.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

        u32 physical_device_count = 0;
        vk::resultCheck(instance.enumeratePhysicalDevices(&physical_device_count, nullptr), "Failed to enumerate physical devices");
        std::vector<vk::PhysicalDevice> physical_devices(physical_device_count);
        vk::resultCheck(instance.enumeratePhysicalDevices(&physical_device_count, physical_devices.data()), "Failed to enumerate physical devices");

        physical_device = physical_devices[0];

        u32 queue_family_count = 0;
        physical_device.getQueueFamilyProperties(&queue_family_count, nullptr);
        std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
        physical_device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

        graphics_queue_family_index = std::numeric_limits<u32>::max();
        for (u32 i = 0; i < queue_families.size(); i++) {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphics_queue_family_index = i;
                break;
            }
        }

        std::array queue_priorities = {1.0f};

        vk::DeviceQueueCreateInfo queue_info = {};
        queue_info.queueFamilyIndex = graphics_queue_family_index;
        queue_info.queueCount = queue_priorities.size();
        queue_info.pQueuePriorities = queue_priorities.data();

        vk::PhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features = {};
        vk::PhysicalDeviceSynchronization2Features synchronization2_features = {};
        synchronization2_features.pNext = &dynamic_rendering_features;

        vk::PhysicalDeviceFeatures2 device_features2 = {};
        device_features2.pNext = &synchronization2_features;
        physical_device.getFeatures2(&device_features2);

        std::vector<const char*> device_extensions = {};
        device_extensions.push_back("VK_KHR_swapchain");
        device_extensions.push_back("VK_KHR_synchronization2");
        device_extensions.push_back("VK_KHR_dynamic_rendering");
        device_extensions.push_back("VK_KHR_portability_subset");

        vk::DeviceCreateInfo device_info = {};
        device_info.pNext = &device_features2;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = device_extensions.size();
        device_info.ppEnabledExtensionNames = device_extensions.data();

        vk::resultCheck(physical_device.createDevice(&device_info, nullptr, &logical_device), "Failed to create logical device");
        vk::defaultDispatchLoaderDynamic.init(logical_device);

        logical_device.getQueue(graphics_queue_family_index, 0, &graphics_queue);
    }

    void destroyLogicalDevice() {
        logical_device.destroy();
    }

    void initSwapchain() {
        u32 surface_format_count = 0;
        vk::resultCheck(physical_device.getSurfaceFormatsKHR(surface, &surface_format_count, nullptr), "Failed to get surface formats");
        std::vector<vk::SurfaceFormatKHR> surface_formats(surface_format_count);
        vk::resultCheck(physical_device.getSurfaceFormatsKHR(surface, &surface_format_count, surface_formats.data()), "Failed to get surface formats");

        surface_format = surface_formats[0];

        u32 present_mode_count = 0;
        vk::resultCheck(physical_device.getSurfacePresentModesKHR(surface, &present_mode_count, nullptr), "Failed to get surface present modes");
        std::vector<vk::PresentModeKHR> present_modes(present_mode_count);
        vk::resultCheck(physical_device.getSurfacePresentModesKHR(surface, &present_mode_count, present_modes.data()), "Failed to get surface present modes");

        vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;

        vk::SurfaceCapabilitiesKHR surface_capabilities;
        vk::resultCheck(physical_device.getSurfaceCapabilitiesKHR(surface, &surface_capabilities), "Failed to get surface capabilities");

        swapchain_extent = surface_capabilities.currentExtent;

        u32 image_count = surface_capabilities.minImageCount + 1;
        if (surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount) {
            image_count = surface_capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR swapchain_info = {};
        swapchain_info.surface = surface;
        swapchain_info.minImageCount = image_count;
        swapchain_info.imageFormat = surface_format.format;
        swapchain_info.imageColorSpace = surface_format.colorSpace;
        swapchain_info.imageExtent = swapchain_extent;
        swapchain_info.imageArrayLayers = 1;
        swapchain_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        swapchain_info.imageSharingMode = vk::SharingMode::eExclusive;
        swapchain_info.preTransform = surface_capabilities.currentTransform;
        swapchain_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        swapchain_info.presentMode = present_mode;
        swapchain_info.clipped = VK_TRUE;
        swapchain_info.oldSwapchain = nullptr;

        vk::resultCheck(logical_device.createSwapchainKHR(&swapchain_info, nullptr, &swapchain), "Failed to create swapchain");

        u32 swapchain_image_count;
        vk::resultCheck(logical_device.getSwapchainImagesKHR(swapchain, &swapchain_image_count, nullptr), "Failed to get swapchain images");
        swapchain_images.resize(swapchain_image_count);
        vk::resultCheck(logical_device.getSwapchainImagesKHR(swapchain, &swapchain_image_count, swapchain_images.data()), "Failed to get swapchain images");

        swapchain_image_views.resize(swapchain_image_count);
        for (u32 i = 0; i < swapchain_image_count; ++i) {
            vk::ImageViewCreateInfo image_view_info = {};
            image_view_info.image = swapchain_images[i];
            image_view_info.viewType = vk::ImageViewType::e2D;
            image_view_info.format = surface_format.format;
            image_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            image_view_info.subresourceRange.baseMipLevel = 0;
            image_view_info.subresourceRange.levelCount = 1;
            image_view_info.subresourceRange.baseArrayLayer = 0;
            image_view_info.subresourceRange.layerCount = 1;

            vk::resultCheck(logical_device.createImageView(&image_view_info, nullptr, &swapchain_image_views[i]), "Failed to create image view");
        }

        depth_images.resize(swapchain_image_count);
        depth_image_views.resize(swapchain_image_count);
        depth_image_memories.resize(swapchain_image_count);
        for (u32 i = 0; i < swapchain_image_count; ++i) {
            vk::ImageCreateInfo image_info = {};
            image_info.imageType = vk::ImageType::e2D;
            image_info.extent.width = swapchain_extent.width;
            image_info.extent.height = swapchain_extent.height;
            image_info.extent.depth = 1;
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = vk::Format::eD32Sfloat;
            image_info.tiling = vk::ImageTiling::eOptimal;
            image_info.initialLayout = vk::ImageLayout::eUndefined;
            image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
            image_info.samples = vk::SampleCountFlagBits::e1;
            image_info.sharingMode = vk::SharingMode::eExclusive;

            vk::resultCheck(logical_device.createImage(&image_info, nullptr, &depth_images[i]), "Failed to create depth image");

            vk::MemoryRequirements memory_requirements;
            logical_device.getImageMemoryRequirements(depth_images[i], &memory_requirements);

            vk::MemoryAllocateInfo memory_allocate_info = {};
            memory_allocate_info.allocationSize = memory_requirements.size;
            memory_allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

            vk::resultCheck(logical_device.allocateMemory(&memory_allocate_info, nullptr, &depth_image_memories[i]), "Failed to allocate depth image memory");

            logical_device.bindImageMemory(depth_images[i], depth_image_memories[i], 0);

            vk::ImageViewCreateInfo image_view_info = {};
            image_view_info.image = depth_images[i];
            image_view_info.viewType = vk::ImageViewType::e2D;
            image_view_info.format = vk::Format::eD32Sfloat;
            image_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            image_view_info.subresourceRange.baseMipLevel = 0;
            image_view_info.subresourceRange.levelCount = 1;
            image_view_info.subresourceRange.baseArrayLayer = 0;
            image_view_info.subresourceRange.layerCount = 1;

            vk::resultCheck(logical_device.createImageView(&image_view_info, nullptr, &depth_image_views[i]), "Failed to create depth image view");
        }
    }

    auto findMemoryType(u32 memoryTypeBits, vk::MemoryPropertyFlags memory_property_flags) -> u32 {
        vk::PhysicalDeviceMemoryProperties memory_properties;
        physical_device.getMemoryProperties(&memory_properties);

        for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & memory_property_flags) == memory_property_flags) {
                return i;
            }
        }
        return std::numeric_limits<u32>::max();
    }

    void destroySwapchain() {
        for (auto& image_view : swapchain_image_views) {
            logical_device.destroyImageView(image_view);
        }
        for (auto& image_view : depth_image_views) {
            logical_device.destroyImageView(image_view);
        }
        for (auto& image_memory : depth_image_memories) {
            logical_device.freeMemory(image_memory);
        }
        for (auto& image : depth_images) {
            logical_device.destroyImage(image);
        }
        logical_device.destroySwapchainKHR(swapchain);
    }

    void initSyncObjects() {
        vk::FenceCreateInfo fence_info = {};
        fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

        vk::SemaphoreCreateInfo semaphore_info = {};

        in_flight_fences.resize(max_frames_in_flight);
        image_available_semaphores.resize(max_frames_in_flight);
        render_finished_semaphores.resize(max_frames_in_flight);

        for (u32 i = 0; i < max_frames_in_flight; ++i) {
            vk::resultCheck(logical_device.createFence(&fence_info, nullptr, &in_flight_fences[i]), "Failed to create fence");
            vk::resultCheck(logical_device.createSemaphore(&semaphore_info, nullptr, &image_available_semaphores[i]), "Failed to create semaphore");
            vk::resultCheck(logical_device.createSemaphore(&semaphore_info, nullptr, &render_finished_semaphores[i]), "Failed to create semaphore");
        }
    }

    void destroySyncObjects() {
        for (u32 i = 0; i < max_frames_in_flight; ++i) {
            logical_device.destroySemaphore(render_finished_semaphores[i]);
            logical_device.destroySemaphore(image_available_semaphores[i]);
            logical_device.destroyFence(in_flight_fences[i]);
        }
    }

    void initCommandPool() {
        vk::CommandPoolCreateInfo command_pool_info = {};
        command_pool_info.queueFamilyIndex = graphics_queue_family_index;
        command_pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

        vk::resultCheck(logical_device.createCommandPool(&command_pool_info, nullptr, &command_pool), "Failed to create command pool");
    }

    void destroyCommandPool() {
        logical_device.destroyCommandPool(command_pool);
    }

    void initCommandBuffers() {
        command_buffers.resize(max_frames_in_flight);

        vk::CommandBufferAllocateInfo command_buffer_allocate_info = {};
        command_buffer_allocate_info.commandPool = command_pool;
        command_buffer_allocate_info.level = vk::CommandBufferLevel::ePrimary;
        command_buffer_allocate_info.commandBufferCount = max_frames_in_flight;
        vk::resultCheck(logical_device.allocateCommandBuffers(&command_buffer_allocate_info, command_buffers.data()), "Failed to allocate command buffers");
    }

    void destroyCommandBuffers() {
        logical_device.freeCommandBuffers(command_pool, max_frames_in_flight, command_buffers.data());
    }

    void initDescriptorPool() {
        vk::DescriptorPoolSize pool_sizes[1];
        pool_sizes[0].type = vk::DescriptorType::eCombinedImageSampler;
        pool_sizes[0].descriptorCount = 1;

        vk::DescriptorPoolCreateInfo pool_info = {};
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = pool_sizes;
        pool_info.maxSets = 1;

        vk::resultCheck(logical_device.createDescriptorPool(&pool_info, nullptr, &descriptor_pool), "Failed to create descriptor pool");
    }

    void destroyDescriptorPool() {
        logical_device.destroyDescriptorPool(descriptor_pool);
    }

    void initGraphicsPipelines() {
        vk::DescriptorSetLayoutBinding layout_bindings[1];
        layout_bindings[0].binding = 0;
        layout_bindings[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        layout_bindings[0].descriptorCount = 1;
        layout_bindings[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
        layout_bindings[0].pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.bindingCount = 1;
        layout_info.pBindings = layout_bindings;

        vk::resultCheck(logical_device.createDescriptorSetLayout(&layout_info, nullptr, &unlit_normal_material.descriptor_set_layout), "Failed to create descriptor set layout");

        vk::PushConstantRange push_constant_range = {};
        push_constant_range.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(ShaderUniforms);

        vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &unlit_normal_material.descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        vk::resultCheck(logical_device.createPipelineLayout(&pipeline_layout_info, nullptr, &unlit_normal_material.pipeline_layout), "Failed to create pipeline layout");

        vk::ShaderModule vert_shader_module = createShaderModule("shaders/Unlit-Normal.vert.spv");
        vk::ShaderModule frag_shader_module = createShaderModule("shaders/Unlit-Normal.frag.spv");

        vk::PipelineShaderStageCreateInfo shader_stages[2] = {};
        shader_stages[0].stage = vk::ShaderStageFlagBits::eVertex;
        shader_stages[0].module = vert_shader_module;
        shader_stages[0].pName = "main";
        shader_stages[1].stage = vk::ShaderStageFlagBits::eFragment;
        shader_stages[1].module = frag_shader_module;
        shader_stages[1].pName = "main";

        vk::VertexInputAttributeDescription attributes[] = {
            {0, 0, vk::Format::eR32G32B32Sfloat, 0},
            {1, 0, vk::Format::eR32G32B32Sfloat, 12}
        };

        vk::VertexInputBindingDescription bindings[] = {
            {0, sizeof(f32) * 6, vk::VertexInputRate::eVertex}
        };

        vk::PipelineVertexInputStateCreateInfo vertex_input_info = {};
//        vertex_input_info.vertexBindingDescriptionCount = 1;
//        vertex_input_info.pVertexBindingDescriptions = bindings;
//        vertex_input_info.vertexAttributeDescriptionCount = 2;
//        vertex_input_info.pVertexAttributeDescriptions = attributes;

        vk::PipelineInputAssemblyStateCreateInfo input_assembly_info = {};
        input_assembly_info.topology = vk::PrimitiveTopology::eTriangleList;
        input_assembly_info.primitiveRestartEnable = VK_FALSE;

        vk::PipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer_info = {};
//        rasterizer_info.cullMode = vk::CullModeFlagBits::eBack;
//        rasterizer_info.frontFace = vk::FrontFace::eClockwise;
        rasterizer_info.lineWidth = 1.0f;
        rasterizer_info.polygonMode = vk::PolygonMode::eFill;

        vk::PipelineMultisampleStateCreateInfo multisampling_info = {};

        vk::PipelineColorBlendAttachmentState color_blend_attachment = {};
        color_blend_attachment.colorWriteMask |= vk::ColorComponentFlagBits::eR;
        color_blend_attachment.colorWriteMask |= vk::ColorComponentFlagBits::eG;
        color_blend_attachment.colorWriteMask |= vk::ColorComponentFlagBits::eB;
        color_blend_attachment.colorWriteMask |= vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo color_blending_info = {};
        color_blending_info.attachmentCount = 1;
        color_blending_info.pAttachments = &color_blend_attachment;

        vk::PipelineDepthStencilStateCreateInfo depth_stencil_info = {};
        depth_stencil_info.depthTestEnable = VK_TRUE;
        depth_stencil_info.depthWriteEnable = VK_TRUE;
        depth_stencil_info.depthCompareOp = vk::CompareOp::eLess;
        depth_stencil_info.minDepthBounds = 0.0f;
        depth_stencil_info.maxDepthBounds = 1.0f;

        vk::DynamicState dynamic_states[] = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamic_state = {};
        dynamic_state.dynamicStateCount = 2;
        dynamic_state.pDynamicStates = dynamic_states;

        vk::Format color_attachment_format = surface_format.format;
        vk::Format depth_attachment_format = vk::Format::eD32Sfloat;

        vk::PipelineRenderingCreateInfo rendering_info = {};
        rendering_info.pNext = nullptr;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachmentFormats = &color_attachment_format;
        rendering_info.depthAttachmentFormat = depth_attachment_format;

        vk::GraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.pNext = &rendering_info;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages;
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly_info;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer_info;
        pipeline_info.pMultisampleState = &multisampling_info;
        pipeline_info.pColorBlendState = &color_blending_info;
        pipeline_info.pDepthStencilState = &depth_stencil_info;
        pipeline_info.pDynamicState = &dynamic_state;
        pipeline_info.layout = unlit_normal_material.pipeline_layout;

        vk::resultCheck(logical_device.createGraphicsPipelines(nullptr, 1, &pipeline_info, nullptr, &unlit_normal_material.graphics_pipeline), "Failed to create graphics pipeline");

        vk::DescriptorSetAllocateInfo descriptor_set_allocate_info = {};
        descriptor_set_allocate_info.descriptorPool = descriptor_pool;
        descriptor_set_allocate_info.descriptorSetCount = 1;
        descriptor_set_allocate_info.pSetLayouts = &unlit_normal_material.descriptor_set_layout;

        vk::resultCheck(logical_device.allocateDescriptorSets(&descriptor_set_allocate_info, &unlit_normal_material.descriptor_set), "Failed to allocate descriptor set");

    }

    void destroyGraphicsPipelines() {
        logical_device.destroyPipeline(unlit_normal_material.graphics_pipeline);
        logical_device.destroyPipelineLayout(unlit_normal_material.pipeline_layout);
    }

    int bufferRowLength = 48;
    int bufferImageHeight = 112;

    void initCarVolumeTexture() {
        FILE* file = fopen("car-volume.png", "rb");
        i32 total_width, total_height;
        stbi_uc* raw = stbi_load_from_file(file, &total_width, &total_height, nullptr, 1);
        fclose(file);

        u32 arrayLayers = 12;
        u32 width = total_width / arrayLayers;
        u32 height = total_height;

        // TextureArray
        vk::ImageCreateInfo image_info = {};
        image_info.imageType = vk::ImageType::e2D;
        image_info.extent.width = width;
        image_info.extent.height = height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = arrayLayers;
        image_info.format = vk::Format::eR8Unorm;
        image_info.tiling = vk::ImageTiling::eOptimal;
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        image_info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        image_info.samples = vk::SampleCountFlagBits::e1;
        image_info.sharingMode = vk::SharingMode::eExclusive;

        vk::resultCheck(logical_device.createImage(&image_info, nullptr, &car_volume_texture.image), "Failed to create image");

        vk::MemoryRequirements image_memory_requirements;
        logical_device.getImageMemoryRequirements(car_volume_texture.image, &image_memory_requirements);

        vk::MemoryAllocateInfo memory_allocate_info = {};
        memory_allocate_info.allocationSize = image_memory_requirements.size;
        memory_allocate_info.memoryTypeIndex = findMemoryType(image_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::resultCheck(logical_device.allocateMemory(&memory_allocate_info, nullptr, &car_volume_texture.memory), "Failed to allocate memory");
        logical_device.bindImageMemory(car_volume_texture.image, car_volume_texture.memory, 0);

        // TextureArrayView
        vk::ImageViewCreateInfo image_view_info = {};
        image_view_info.image = car_volume_texture.image;
        image_view_info.viewType = vk::ImageViewType::e2DArray;
        image_view_info.format = vk::Format::eR8Unorm;
        image_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_view_info.subresourceRange.baseMipLevel = 0;
        image_view_info.subresourceRange.levelCount = 1;
        image_view_info.subresourceRange.baseArrayLayer = 0;
        image_view_info.subresourceRange.layerCount = arrayLayers;

        vk::resultCheck(logical_device.createImageView(&image_view_info, nullptr, &car_volume_texture.view), "Failed to create image view");

        // TextureSampler
        vk::SamplerCreateInfo sampler_info = {};
        sampler_info.magFilter = vk::Filter::eNearest;
        sampler_info.minFilter = vk::Filter::eNearest;
        sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        sampler_info.anisotropyEnable = VK_FALSE;
        sampler_info.maxAnisotropy = 1;
        sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = VK_FALSE;
        sampler_info.compareOp = vk::CompareOp::eAlways;
        sampler_info.mipmapMode = vk::SamplerMipmapMode::eNearest;
        sampler_info.mipLodBias = 0.0f;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = 0.0f;

        vk::resultCheck(logical_device.createSampler(&sampler_info, nullptr, &car_volume_texture.sampler), "Failed to create sampler");

        // Upload
        vk::Buffer staging_buffer;
        vk::BufferCreateInfo staging_buffer_info = {};
        staging_buffer_info.size = total_width * total_height;
        staging_buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        staging_buffer_info.sharingMode = vk::SharingMode::eExclusive;
        vk::resultCheck(logical_device.createBuffer(&staging_buffer_info, nullptr, &staging_buffer), "Failed to create staging buffer");

        vk::MemoryRequirements staging_buffer_memory_requirements;
        logical_device.getBufferMemoryRequirements(staging_buffer, &staging_buffer_memory_requirements);

        vk::MemoryAllocateInfo staging_buffer_memory_allocate_info = {};
        staging_buffer_memory_allocate_info.allocationSize = staging_buffer_memory_requirements.size;
        staging_buffer_memory_allocate_info.memoryTypeIndex = findMemoryType(staging_buffer_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        vk::DeviceMemory staging_buffer_memory;
        vk::resultCheck(logical_device.allocateMemory(&staging_buffer_memory_allocate_info, nullptr, &staging_buffer_memory), "Failed to allocate staging buffer memory");

        logical_device.bindBufferMemory(staging_buffer, staging_buffer_memory, 0);

        void* data;
        vk::resultCheck(logical_device.mapMemory(staging_buffer_memory, 0, staging_buffer_memory_requirements.size, vk::MemoryMapFlags(), &data), "Failed to map memory");
        std::memcpy(data, raw, width * height);
        logical_device.unmapMemory(staging_buffer_memory);

        vk::CommandBuffer command_buffer = createCommandBuffer(command_pool, vk::CommandBufferLevel::ePrimary);

        vk::CommandBufferBeginInfo command_buffer_begin_info = {};
        command_buffer_begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        vk::resultCheck(command_buffer.begin(&command_buffer_begin_info), "Failed to begin command buffer");

        vk::ImageMemoryBarrier2 image_memory_barrier_1 = {};
        image_memory_barrier_1.oldLayout = vk::ImageLayout::eUndefined;
        image_memory_barrier_1.newLayout = vk::ImageLayout::eTransferDstOptimal;
        image_memory_barrier_1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier_1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier_1.image = car_volume_texture.image;
        image_memory_barrier_1.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_memory_barrier_1.subresourceRange.baseMipLevel = 0;
        image_memory_barrier_1.subresourceRange.levelCount = 1;
        image_memory_barrier_1.subresourceRange.baseArrayLayer = 0;
        image_memory_barrier_1.subresourceRange.layerCount = arrayLayers;
        image_memory_barrier_1.srcAccessMask = vk::AccessFlags2();
        image_memory_barrier_1.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        image_memory_barrier_1.srcStageMask = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
        image_memory_barrier_1.dstStageMask = vk::PipelineStageFlagBits2KHR::eTransfer;

        vk::DependencyInfo dependency_info_1 = {};
        dependency_info_1.imageMemoryBarrierCount = 1;
        dependency_info_1.pImageMemoryBarriers = &image_memory_barrier_1;

        command_buffer.pipelineBarrier2KHR(&dependency_info_1);

        vk::BufferImageCopy buffer_image_copies[12] = {};
        for (u32 i = 0; i < 12; ++i) {
            // Copy only the first layer of the image
            buffer_image_copies[i].bufferOffset = 0;
            buffer_image_copies[i].bufferRowLength = bufferRowLength;
            buffer_image_copies[i].bufferImageHeight = bufferImageHeight;
            buffer_image_copies[i].imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            buffer_image_copies[i].imageSubresource.mipLevel = 0;
            buffer_image_copies[i].imageSubresource.baseArrayLayer = i;
            buffer_image_copies[i].imageSubresource.layerCount = 1;
            buffer_image_copies[i].imageOffset = vk::Offset3D(0, 0, 0);
            buffer_image_copies[i].imageExtent = vk::Extent3D(width, height, 1);
        }

        command_buffer.copyBufferToImage(staging_buffer, car_volume_texture.image, vk::ImageLayout::eTransferDstOptimal, 12, buffer_image_copies);

        vk::ImageMemoryBarrier2 image_memory_barrier_2 = {};
        image_memory_barrier_2.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        image_memory_barrier_2.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        image_memory_barrier_2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier_2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier_2.image = car_volume_texture.image;
        image_memory_barrier_2.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_memory_barrier_2.subresourceRange.baseMipLevel = 0;
        image_memory_barrier_2.subresourceRange.levelCount = 1;
        image_memory_barrier_2.subresourceRange.baseArrayLayer = 0;
        image_memory_barrier_2.subresourceRange.layerCount = arrayLayers;
        image_memory_barrier_2.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        image_memory_barrier_2.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        image_memory_barrier_2.srcStageMask = vk::PipelineStageFlagBits2KHR::eTransfer;
        image_memory_barrier_2.dstStageMask = vk::PipelineStageFlagBits2KHR::eFragmentShader;

        vk::DependencyInfo dependency_info_2 = {};
        dependency_info_2.imageMemoryBarrierCount = 1;
        dependency_info_2.pImageMemoryBarriers = &image_memory_barrier_2;

        command_buffer.pipelineBarrier2KHR(&dependency_info_2);
        command_buffer.end();

        submitCommandBuffer(command_buffer);
        logical_device.freeCommandBuffers(command_pool, 1, &command_buffer);
        stbi_image_free(raw);

        vk::DescriptorImageInfo descriptor_image_info = {};
        descriptor_image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        descriptor_image_info.imageView = car_volume_texture.view;
        descriptor_image_info.sampler = car_volume_texture.sampler;

        vk::WriteDescriptorSet write_descriptor_set = {};
        write_descriptor_set.dstSet = unlit_normal_material.descriptor_set;
        write_descriptor_set.dstBinding = 0;
        write_descriptor_set.dstArrayElement = 0;
        write_descriptor_set.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        write_descriptor_set.descriptorCount = 1;
        write_descriptor_set.pImageInfo = &descriptor_image_info;

        logical_device.updateDescriptorSets(1, &write_descriptor_set, 0, nullptr);
    }

    void destroyCarVolumeTexture() {

    }

    auto createCommandBuffer(vk::CommandPool pool, vk::CommandBufferLevel level) -> vk::CommandBuffer {
        vk::CommandBufferAllocateInfo command_buffer_allocate_info = {};
        command_buffer_allocate_info.commandPool = pool;
        command_buffer_allocate_info.level = level;
        command_buffer_allocate_info.commandBufferCount = 1;

        vk::CommandBuffer command_buffer;
        vk::resultCheck(logical_device.allocateCommandBuffers(&command_buffer_allocate_info, &command_buffer), "Failed to allocate command buffer");
        return command_buffer;
    }

    void submitCommandBuffer(vk::CommandBuffer command_buffer) {
        vk::SubmitInfo submit_info = {};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vk::Fence fence;
        vk::FenceCreateInfo fence_info = {};
        vk::resultCheck(logical_device.createFence(&fence_info, nullptr, &fence), "Failed to create fence");
        vk::resultCheck(graphics_queue.submit(1, &submit_info, fence), "Failed to submit command buffer");
        vk::resultCheck(logical_device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX), "Failed to wait for fence");
        logical_device.destroyFence(fence);
    }

    auto createShaderModule(const std::string& filename) -> vk::ShaderModule {
        auto [it, _] = shader_modules.emplace(filename, lazy{[&] {
            fprintf(stdout, "Loading shader module: %s\n", filename.c_str());

            auto bytes = readFileBytes(filename);

            vk::ShaderModuleCreateInfo create_info = {};
            create_info.codeSize = bytes.size();
            create_info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

            vk::ShaderModule shader_module;
            vk::resultCheck(logical_device.createShaderModule(&create_info, nullptr, &shader_module), "Failed to create shader module");
            return shader_module;
        }});
        return it->second;
    }

    void destroyShaderModules() {
        for (auto& [_, shader_module] : shader_modules) {
            logical_device.destroyShaderModule(shader_module);
        }
        shader_modules.clear();
    }

    void startEventLoop() {
        auto keys = SDL_GetKeyboardState(nullptr);
        SDL_SetRelativeMouseMode(SDL_TRUE);

        auto previous = std::chrono::steady_clock::now();

        bool quit = false;
        while (!quit) {
            using secongs = std::chrono::duration<f32, std::chrono::seconds::period>;
            auto current = std::chrono::steady_clock::now();
            auto elapsed = secongs(current - previous).count();
            previous = current;

            glm::vec3 input_direction = glm::vec3(0, 0, 0);

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT: {
                        quit = true;
                        break;
                    }
                    case SDL_KEYDOWN: {
                        if (event.key.keysym.sym == SDLK_q) {
                            logical_device.waitIdle();
                            bufferRowLength += 1;
                            fprintf(stdout, "BufferRowLength: %d\n", bufferRowLength);
                            initCarVolumeTexture();
                        }
                        if (event.key.keysym.sym == SDLK_e) {
                            logical_device.waitIdle();
                            bufferRowLength -= 1;
                            fprintf(stdout, "BufferRowLength: %d\n", bufferRowLength);
                            initCarVolumeTexture();
                        }
                        if (event.key.keysym.sym == SDLK_r) {
                            logical_device.waitIdle();
                            bufferImageHeight += 1;
                            fprintf(stdout, "BufferImageHeight: %d\n", bufferImageHeight);
                            initCarVolumeTexture();
                        }
                        if (event.key.keysym.sym == SDLK_t) {
                            logical_device.waitIdle();
                            bufferImageHeight -= 1;
                            fprintf(stdout, "BufferImageHeight: %d\n", bufferImageHeight);
                            initCarVolumeTexture();
                        }
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }

            if (keys[SDL_SCANCODE_W]) {
                input_direction += glm::vec3(0, 0, 1);
            }
            if (keys[SDL_SCANCODE_S]) {
                input_direction += glm::vec3(0, 0, -1);
            }
            if (keys[SDL_SCANCODE_A]) {
                input_direction += glm::vec3(-1, 0, 0);
            }
            if (keys[SDL_SCANCODE_D]) {
                input_direction += glm::vec3(1, 0, 0);
            }

            if (input_direction != glm::vec3(0, 0, 0)) {
                input_direction = glm::normalize(input_direction);
            }

            i32 dx, dy;
            SDL_GetRelativeMouseState(&dx, &dy);

            camera_rotation.y += static_cast<f32>(dx) * elapsed * 0.25F;
            camera_rotation.x += static_cast<f32>(dy) * elapsed * 0.25F;

            camera_position += glm::mat3(glm::quat(camera_rotation)) * input_direction * elapsed * 10.0F;

            vk::resultCheck(logical_device.waitForFences(1, &in_flight_fences[frame_index], VK_TRUE, UINT64_MAX), "Failed to wait for fences");
            vk::resultCheck(logical_device.resetFences(1, &in_flight_fences[frame_index]), "Failed to reset fences");
            vk::resultCheck(logical_device.acquireNextImageKHR(swapchain, UINT64_MAX, image_available_semaphores[frame_index], nullptr, &image_index), "Failed to acquire next image");

            recordCommandBuffer();

            vk::PipelineStageFlags wait_stages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

            vk::SubmitInfo submit_info = {};
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &image_available_semaphores[frame_index];
            submit_info.pWaitDstStageMask = &wait_stages;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffers[image_index];
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &render_finished_semaphores[frame_index];

            vk::resultCheck(graphics_queue.submit(1, &submit_info, in_flight_fences[frame_index]), "Failed to submit draw command buffer");

            vk::PresentInfoKHR present_info = {};
            present_info.waitSemaphoreCount = 1;
            present_info.pWaitSemaphores = &render_finished_semaphores[frame_index];
            present_info.swapchainCount = 1;
            present_info.pSwapchains = &swapchain;
            present_info.pImageIndices = &image_index;

            vk::resultCheck(graphics_queue.presentKHR(&present_info), "Failed to present");

            frame_index = (frame_index + 1) % max_frames_in_flight;
        }
    }

    void recordCommandBuffer() {
        vk::CommandBufferBeginInfo command_buffer_begin_info = {};
        command_buffer_begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        vk::resultCheck(command_buffers[frame_index].begin(&command_buffer_begin_info), "Failed to begin recording command buffer");

        // Change image layout to color attachment
        vk::ImageMemoryBarrier2 image_memory_barriers_1[2] = {};
        image_memory_barriers_1[0].srcStageMask = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
        image_memory_barriers_1[0].dstStageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput;
        image_memory_barriers_1[0].srcAccessMask = vk::AccessFlagBits2KHR::eNone;
        image_memory_barriers_1[0].dstAccessMask = vk::AccessFlagBits2KHR::eColorAttachmentRead | vk::AccessFlagBits2KHR::eColorAttachmentWrite;
        image_memory_barriers_1[0].oldLayout = vk::ImageLayout::eUndefined;
        image_memory_barriers_1[0].newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        image_memory_barriers_1[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_1[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_1[0].image = swapchain_images[image_index];
        image_memory_barriers_1[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_memory_barriers_1[0].subresourceRange.baseMipLevel = 0;
        image_memory_barriers_1[0].subresourceRange.levelCount = 1;
        image_memory_barriers_1[0].subresourceRange.baseArrayLayer = 0;
        image_memory_barriers_1[0].subresourceRange.layerCount = 1;

        image_memory_barriers_1[1].srcStageMask = vk::PipelineStageFlagBits2KHR::eTopOfPipe;
        image_memory_barriers_1[1].dstStageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests | vk::PipelineStageFlagBits2KHR::eLateFragmentTests;
        image_memory_barriers_1[1].srcAccessMask = vk::AccessFlagBits2KHR::eNone;
        image_memory_barriers_1[1].dstAccessMask = vk::AccessFlagBits2KHR::eDepthStencilAttachmentRead | vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite;
        image_memory_barriers_1[1].oldLayout = vk::ImageLayout::eUndefined;
        image_memory_barriers_1[1].newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        image_memory_barriers_1[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_1[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_1[1].image = depth_images[image_index];
        image_memory_barriers_1[1].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        image_memory_barriers_1[1].subresourceRange.baseMipLevel = 0;
        image_memory_barriers_1[1].subresourceRange.levelCount = 1;
        image_memory_barriers_1[1].subresourceRange.baseArrayLayer = 0;
        image_memory_barriers_1[1].subresourceRange.layerCount = 1;

        vk::DependencyInfo dependency_info_1 = {};
        dependency_info_1.dependencyFlags = vk::DependencyFlagBits::eByRegion;
        dependency_info_1.imageMemoryBarrierCount = 2;
        dependency_info_1.pImageMemoryBarriers = image_memory_barriers_1;

        command_buffers[frame_index].pipelineBarrier2KHR(&dependency_info_1);

        vk::Rect2D render_area = {};
        render_area.offset.x = 0;
        render_area.offset.y = 0;
        render_area.extent.width = swapchain_extent.width;
        render_area.extent.height = swapchain_extent.height;

        vk::RenderingAttachmentInfo color_attachment = {};
        color_attachment.imageView = swapchain_image_views[image_index];
        color_attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
        color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
        color_attachment.clearValue.color.float32[0] = 0.0f;
        color_attachment.clearValue.color.float32[1] = 0.0f;
        color_attachment.clearValue.color.float32[2] = 0.0f;
        color_attachment.clearValue.color.float32[3] = 1.0f;

        vk::RenderingAttachmentInfo depth_attachment = {};
        depth_attachment.imageView = depth_image_views[image_index];
        depth_attachment.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
        depth_attachment.storeOp = vk::AttachmentStoreOp::eStore;
        depth_attachment.clearValue.depthStencil.depth = 1.0f;
        depth_attachment.clearValue.depthStencil.stencil = 0;

        vk::RenderingInfo rendering_info = {};
        rendering_info.renderArea = render_area;
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        rendering_info.pDepthAttachment = &depth_attachment;

        vk::Viewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<f32>(swapchain_extent.width);
        viewport.height = static_cast<f32>(swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        auto trs = glm::translate(glm::mat4(1.0f), camera_position) * glm::mat4(glm::quat(camera_rotation));
        auto proj = glm::perspectiveLH_ZO(
            -glm::radians(45.0f),
            -static_cast<f32>(swapchain_extent.width) / static_cast<f32>(swapchain_extent.height),
            0.1f,
            100.0f
        );

        ShaderUniforms shader_uniforms = {};
        shader_uniforms.world_to_clip = proj * glm::inverse(trs);
        shader_uniforms.camera_position = camera_position;
//        shader_uniforms.world_to_clip[1][1] *= -1;

        command_buffers[frame_index].beginRendering(&rendering_info);
        command_buffers[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics, unlit_normal_material.graphics_pipeline);
        command_buffers[frame_index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, unlit_normal_material.pipeline_layout, 0, 1, &unlit_normal_material.descriptor_set, 0, nullptr);
        command_buffers[frame_index].pushConstants(unlit_normal_material.pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(ShaderUniforms), &shader_uniforms);
        command_buffers[frame_index].setViewport(0, 1, &viewport);
        command_buffers[frame_index].setScissor(0, 1, &render_area);
        command_buffers[frame_index].draw(36, 1, 0, 0);
        command_buffers[frame_index].endRendering();

        // Change image layout to present
        vk::ImageMemoryBarrier2 image_memory_barriers_2[1] = {};
        image_memory_barriers_2[0].srcStageMask = vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput;
        image_memory_barriers_2[0].dstStageMask = vk::PipelineStageFlagBits2KHR::eBottomOfPipe;
        image_memory_barriers_2[0].srcAccessMask = vk::AccessFlagBits2KHR::eColorAttachmentRead | vk::AccessFlagBits2KHR::eColorAttachmentWrite;
        image_memory_barriers_2[0].dstAccessMask = vk::AccessFlagBits2KHR::eNone;
        image_memory_barriers_2[0].oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        image_memory_barriers_2[0].newLayout = vk::ImageLayout::ePresentSrcKHR;
        image_memory_barriers_2[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_2[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barriers_2[0].image = swapchain_images[image_index];
        image_memory_barriers_2[0].subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_memory_barriers_2[0].subresourceRange.baseMipLevel = 0;
        image_memory_barriers_2[0].subresourceRange.levelCount = 1;
        image_memory_barriers_2[0].subresourceRange.baseArrayLayer = 0;
        image_memory_barriers_2[0].subresourceRange.layerCount = 1;

//        image_memory_barriers_2[1].srcStageMask = vk::PipelineStageFlagBits2KHR::eEarlyFragmentTests | vk::PipelineStageFlagBits2KHR::eLateFragmentTests;
//        image_memory_barriers_2[1].dstStageMask = vk::PipelineStageFlagBits2KHR::eBottomOfPipe;
//        image_memory_barriers_2[1].srcAccessMask = vk::AccessFlagBits2KHR::eDepthStencilAttachmentRead | vk::AccessFlagBits2KHR::eDepthStencilAttachmentWrite;
//        image_memory_barriers_2[1].dstAccessMask = vk::AccessFlagBits2KHR::eNone;
//        image_memory_barriers_2[1].oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
//        image_memory_barriers_2[1].newLayout = vk::ImageLayout::e;
//        image_memory_barriers_2[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vk::DependencyInfo dependency_info_2 = {};
        dependency_info_2.dependencyFlags = vk::DependencyFlagBits::eByRegion;
        dependency_info_2.imageMemoryBarrierCount = 1;
        dependency_info_2.pImageMemoryBarriers = image_memory_barriers_2;

        command_buffers[frame_index].pipelineBarrier2KHR(&dependency_info_2);
        command_buffers[frame_index].end();
    }

    static auto readFileBytes(const std::string& filename) -> std::vector<char> {
        std::ifstream file(filename, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), {}};
    }
};

auto main() -> i32 {
    try {
        VulkanApplication app;
        app.startEventLoop();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
