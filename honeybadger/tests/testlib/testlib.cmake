file(GLOB_RECURSE testlib_src_files CONFIGURE_DEPENDS honeybadger/tests/testlib/src/*.cu)
add_library(testlib STATIC ${testlib_src_files})
target_link_libraries(testlib ${PROJECT_NAME})
target_include_directories(testlib PUBLIC honeybadger/include honeybadger/tests/testlib/include ${PROJECT_BINARY_DIR})