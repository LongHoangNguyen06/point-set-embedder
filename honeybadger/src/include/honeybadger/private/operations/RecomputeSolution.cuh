/**
 * Module for:
 * - compute the crossing number of a layout: compute for every two edges if they
 * are visible and cross.
 * - verify correctness of a drawing.
 * - verify completeness of a drawing.
 */
#pragma once

#include <vector>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Problem.cuh"

namespace honeybadger::recompute_solution {
    /**
     * Functions which compute the (m x m) crossing matrix determines if two edge crosses.
     * Two edges cross iff:
     * - they cross in geometrical sense.
     * - and they are both visible.
     *
     * Store the result in:
     * - m_edge_cross_edge
     *
     * Launch (m x m) threads.
     */
    void edgeCrossEdge(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions which count how many crossings every edge has.
     *
     * Store the result in:
     * - m_edge_crossings
     *
     * Launch m threads.
     */
    void edgeCrossings(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions to compute the objective value of the solution.
     * Store the result in:
     * - m_objective_value
     *
     * Launch 1 thread.
     */
    void objectiveValue(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions to compute how many crossings a node has.
     *
     * Store the result in:
     * - m_node_crossings
     *
     * Launch n threads.
     */
    void nodeCrossings(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions to compute the (p x m) matrix determines which point is crossed
     * by which edge.
     *
     * Store the result in:
     * - m_point_crossed_edge
     *
     * Launch (p x m) threads.
     */
    void pointCrossedEdges(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Functions which compute how many times a point was crossed by any edge.
     *
     * Store the result in:
     * - m_point_crossings
     *
     * Launch p blocks, each block for a point.
     */
    void pointCrossings(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Function which determines if a drawing is correct.
     *
     * Store the result in:
     * - m_is_correct
     *
     * Launch n thread.
     */
    void isCorrect(Solution::Data solution, cudaStream_t stream = nullptr);

    /**
     * Function which determines if a drawing is complete.
     *
     * Store the result in:
     * - m_is_complete
     *
     * Launch 1 thread.
     */
    void isComplete(Solution::Data solution, cudaStream_t stream = nullptr);
}