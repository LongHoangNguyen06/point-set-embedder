#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <cinttypes>
#include <cstring>

typedef int64_t int_t;

typedef float float_t;

constexpr int_t NON_VALID_ASSIGNMENT = -1; // Unassigned points or nodes will have this value.

namespace honeybadger {
    /**
     * @return ceil(n / grain)
     */
    template<typename T1, typename T2>
    __host__ __device__ int_t divup(T1 n, T2 m) {
        return ceil(double(n) / double(m));
    }

    /**
     * Test if a vector has duplicate element.
     * @tparam T type with compare operators.
     * @param vec vector of objects
     * @return true if vector has duplicate.
     */
    template<typename T>
    bool hasDuplicate(const std::vector<T> &vec) {
        return std::set < T > (vec.begin(), vec.end()).size() < vec.size();
    }

    /**
     * @return seconds since epoch
     */
    int_t secondsSinceEpoch();

    /**
     * https://stackoverflow.com/a/56107709
     */
    int_t msSinceEpoch();

    /**
     * Throw this exception when an algorithm is stuck and
     * can not find a solution.
     */
    class NotSatisfiableException : public std::runtime_error {
    public:
        [[maybe_unused]] explicit NotSatisfiableException(const std::string &arg);
    };
}

// ##########################################################
// Help macros
// ##########################################################
#define SUMMARY(X) "n=" << (X).n << ", m=" << (X).m << ", p=" << (X).p
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOGGING std::cerr << FILENAME << ":" << __LINE__  << std::string(35 - strlen(FILENAME) - std::to_string(__LINE__).size(), ' ') << " "
#define LOGGING_NL std::cerr << std::endl << FILENAME << ":" << __LINE__ << std::string(35 - strlen(FILENAME) - std::to_string(__LINE__).size(), ' ') << " "
#define robust_cuda_malloc_managed(ptr, n) cudaMallocManaged((void **) &(ptr), sizeof(ptr[0]) * n)
#define robust_cuda_memcpy(src, dst, n) cudaMemcpy(dst, src, sizeof(src[0]) * n, cudaMemcpyHostToHost)

#define throw_error(msg, exception_clazz)            \
  {                                                  \
    std::stringstream ss;                            \
    ss << FILENAME << ":" << __LINE__ << " " << msg; \
    throw exception_clazz(ss.str());                 \
  }

#define post_kernel_invocation_check gpu_error_check(cudaGetLastError())

#define gpu_error_check(ans)                                                   \
  {                                                                          \
    auto code = (ans);                                                       \
    if (code != cudaSuccess) {                                                 \
      LOGGING << cudaGetErrorString(code) << std::endl; \
      throw_error(cudaGetErrorString(code)                                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl, \
                  std::runtime_error);                                          \
    }                                                                        \
  }
