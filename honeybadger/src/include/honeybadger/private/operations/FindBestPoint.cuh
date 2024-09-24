/***
 * Strategy to find a best point to assign a vertex:
 * - Find all points that are not currently assigned to a vertex. Call those points available points.
 * - Filter points, for which an insertion will cause crossing with assigned points.
 *   We are left with possible points.
 * - From the left points, select points which would introduce minimal crossings.
 *   These are the best points.
 */
#pragma once

#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/Utils.cuh"
#include "honeybadger/Geometry.cuh"

namespace honeybadger::find_best_points {
    /**
     * Functions to compute available points and assigned points.
     *
     * Store the results in:
     * - m_available_points_indices
     * - m_num_available_points
     * - m_assigned_points_indices
     * - m_num_assigned_points
     */
    // Spawn a single thread.
    void computeNumAvailableAndAssignedPoints(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions to compute possible points. For every available point,
     * check if the point crosses any assigned point.
     *
     * Store the results in:
     * - m_available_points_cross_assigned_points
     */
    // Each thread block i checks if inserting the vertex
    // to point i would cross any assigned point.
    void computeAvailablePointsCrossAssignedPoints(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

    /**
     * Functions to compute possible points.
     *
     * Store the results in:
     * - m_possible_points_indices
     * - m_num_possible_points
     */
    // Spawn a single thread.
    void computeNumPossiblePoints(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Function to compute how many crossings possible points would introduce.
     *
     * Store the results in:
     * - m_possible_points_crossing_number
     */
    // Spawn a thread block for every possible point.
    void computePossiblePointsCrossingNumber(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

    /**
     * Functions to compute the points of the possible points set, which
     * would minimize the crossings after inserting a vertex.
     */
    // Spawn a single thread.
    void computeNumBestPoints(Solution::Data solution, cudaStream_t stream = nullptr);
}