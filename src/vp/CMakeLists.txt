add_executable(vp)

target_sources(vp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vp.m.cpp
)

target_link_libraries(vp
    PRIVATE
        glfw
        glm::glm
        vulkan-headers::vulkan-headers
        Vulkan::Loader
        project-options
)

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
