enable_testing()
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)

include(honeybadger/tests/testlib/testlib.cmake)

file(GLOB_RECURSE test_files CONFIGURE_DEPENDS honeybadger/tests/Test*.cu)
foreach (file ${test_files})
    get_filename_component(file_name ${file} NAME)
    get_filename_component(test_name ${file} NAME_WE)
    add_executable(${test_name} ${file})
    add_test(NAME ${test_name} COMMAND ${test_name})
    target_link_libraries(${test_name} ${PROJECT_NAME} GTest::gtest_main testlib)
endforeach ()