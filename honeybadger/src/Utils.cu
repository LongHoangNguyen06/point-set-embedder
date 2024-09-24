#include "honeybadger/Utils.cuh"
#include <chrono>

namespace honeybadger {
    int_t secondsSinceEpoch() {
        return static_cast<int_t>(std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    int_t msSinceEpoch() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    [[maybe_unused]] NotSatisfiableException::NotSatisfiableException(const std::string &arg) : runtime_error(arg) {}
}