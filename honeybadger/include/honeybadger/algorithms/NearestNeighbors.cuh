#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include <map>

namespace honeybadger::algorithms::nearest_neighbors{
    /**
     * Nearest neighbor solver.
     *
     * Assign each vertex the nearest point of the vertex' neighbors' points.
     */
    void initialize(Solution::Ptr solution);
}