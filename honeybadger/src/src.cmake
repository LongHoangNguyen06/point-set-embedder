find_package(OpenMP REQUIRED)

# Compile nlohmann_json with C++17
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)
add_library(JSON STATIC honeybadger/src/JSON.cu)
set_target_properties(JSON PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 17)
target_compile_options(JSON PUBLIC ${CMAKE_CXX_FLAGS};-forward-unknown-to-host-compiler) # or-tools stresses
target_include_directories(JSON PUBLIC honeybadger/include ${PROJECT_BINARY_DIR})
target_link_libraries(JSON PUBLIC nlohmann_json::nlohmann_json)

# Compile everything else exception with C++20
file(GLOB_RECURSE src_files CONFIGURE_DEPENDS honeybadger/src/**/*.cu honeybadger/src/*.cu)
list(REMOVE_ITEM src_files ${CMAKE_SOURCE_DIR}/honeybadger/src/JSON.cu)
add_library(${PROJECT_NAME} STATIC ${src_files})
target_compile_options(${PROJECT_NAME} PUBLIC "${CMAKE_CXX_FLAGS};-forward-unknown-to-host-compiler") # or-tools stresses
target_include_directories(${PROJECT_NAME} PUBLIC honeybadger/include ${PROJECT_BINARY_DIR} PRIVATE honeybadger/src/include)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 20)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX JSON) # gcc-10 stresses