#pragma once

#include "honeybadger/honeybadger.cuh"

namespace honeybadger::algorithms::naive_algorithm {
    /**
     * Deterministic to solve an empty drawing.
     *
     * @param solution input operations.
     */
    void init(Solution::Ptr solution);

    /**
     * Deterministic algorithm to improve a complete drawing.
     * @param solution input operations.
     */
    void imp(Solution::Ptr solution);
}