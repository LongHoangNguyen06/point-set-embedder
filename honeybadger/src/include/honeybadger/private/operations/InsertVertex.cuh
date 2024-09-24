#pragma once

#include "honeybadger/Utils.cuh"

namespace honeybadger::operations {
    /**
     * Insert a vertex into the graph.
     * Assume the insertion will keep the drawing valid.
     *
     * @param solution input operations.
     * @param vertex input vertex.
     * @param point_idx at which point should the vertex be inserted.
     *
     * Throw exception if the vertex or the point is already assigned in the operations.
     */
    void insertVertex(Solution::Data solution, int_t vertex, int_t point_idx);

    namespace insert_vertex {
        /**
         * Functions to update the (m x m) crossing matrix after inserting a vertex.
         *
         * Also, update partially m_edge_crossings of edges.
         */
        void edgeCrossEdge(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

        /**
         * Function to update edge crossings after inserting a vertex.
         */
        // This only update the edge crossings of incident edges of the vertex since
        // the other parts were updated in edgeCrossEdge.
        void edgeCrossings(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

        /**
         * Function to compute the objective value after inserting a vertex.
         */
        void objectiveValue(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

        /**
         * Function to update the objective value of every edge after inserting a vertex.
         */
        void nodeCrossings(Solution::Data solution, int_t vertex);

        /**
         * Function to update the crossings of every point after inserting a vertex.
         */
        void pointCrossedEdges(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);
    }
}