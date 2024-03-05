add_executable(vp)

set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

target_sources(vp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vp.m.cpp
    PRIVATE
        ${SHADER_BINARY_DIR}/vert.spv
        ${SHADER_BINARY_DIR}/frag.spv
)

target_link_libraries(vp
    PRIVATE
        glfw
        glm::glm
        vulkan-headers::vulkan-headers
        Vulkan::Loader
        project-options
)

add_custom_command(
    OUTPUT ${SHADER_BINARY_DIR}/vert.spv
    COMMAND 
        ${GLSLC_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert -o ${SHADER_BINARY_DIR}/vert.spv
)

add_custom_command(
    OUTPUT ${SHADER_BINARY_DIR}/frag.spv
    COMMAND 
        ${GLSLC_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag -o ${SHADER_BINARY_DIR}/frag.spv
)

set_property(TARGET vp 
    PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if (VP_BUILD_TESTS)
    add_executable(vp_test)

    target_sources(vp_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vp.t.cpp
    )

    target_link_libraries(vp_test
        PRIVATE
            Catch2::Catch2WithMain
            project-options
    )

    include(Catch)
    catch_discover_tests(vp_test)
endif()
