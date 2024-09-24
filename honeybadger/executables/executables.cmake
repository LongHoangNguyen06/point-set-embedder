FetchContent_Declare(cli11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11 GIT_TAG v2.2.0)
FetchContent_MakeAvailable(cli11)
add_executable(honeybadger
        honeybadger/executables/HoneyBadger.cu
        honeybadger/executables/CLI.cu)
target_link_libraries(honeybadger ${PROJECT_NAME} CLI11)
target_include_directories(honeybadger PUBLIC ${CLI11_DIR}/include)