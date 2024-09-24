#pragma once

#include "honeybadger/datastructure/Solution.cuh"

namespace honeybadger::algorithms::random_search {
    /**
     * Randomized to solve an empty drawing by inserting vertex for vertex greedily.
     *
     * Assigned vertices will be ignored by the algorithm, i.e., an assigned vertex will not be reassigned.
     */
    void initialize(Solution::Ptr solution, int_t seed = 0);

    /**
     * Randomized improve a complete solution.
     *
     * Phase 1: Sort the vertex by crossings and greedily reinsert vertex for vertex. Stop when user interrupts.
     * Phase 2: Local search.
     */
    void improve(Solution::Ptr solution, int_t seed = 0, std::optional<int_t> neighborhood = std::nullopt);

    /**
     * Find a solution from scratch, optimize it until users interrupts.
     *
     * Phase 1: Insert vertex greedily, sorted by degree.
     * Phase 2: Sort the vertex by crossings and greedily reinsert vertex for vertex. Stop when user interrupts.
     * Phase 3: Local search.
     */
    void initializeAndImprove(Solution::Ptr solution, int_t seed = 0, std::optional<int_t> neighborhood = std::nullopt);
}