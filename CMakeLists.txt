cmake_minimum_required(VERSION 3.24)
project(game)

include(FetchContent)

set(CMAKE_CXX_STANDARD 23)

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)

FetchContent_Declare(glm URL "${CMAKE_CURRENT_SOURCE_DIR}/deps/glm-0.9.9.8.zip")
FetchContent_MakeAvailable(glm)

add_executable(game main.cpp stb_image.h)
target_link_libraries(game SDL2::SDL2 Vulkan::Vulkan glm)
target_precompile_headers(game PRIVATE std_core.hpp)

function(target_compile_shaders TARGET)
    set(SPIRV_BINARY_FILES)

    foreach(SOURCE_FILE ${ARGN})
        set(SPIRV "${SOURCE_FILE}.spv")
        add_custom_command(
            OUTPUT ${SPIRV}
                # Enable SPV_KHR_non_semantic_info
            COMMAND glslc ${SOURCE_FILE} -o ${SPIRV}
            DEPENDS ${SOURCE_FILE}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        list(APPEND SPIRV_BINARY_FILES ${SPIRV})
    endforeach()

    add_custom_target(${TARGET}_shaders DEPENDS ${SPIRV_BINARY_FILES})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()

file(GLOB_RECURSE SHADER_FILES
    "shaders/*.frag"
    "shaders/*.vert"
)

target_compile_shaders(game ${SHADER_FILES})