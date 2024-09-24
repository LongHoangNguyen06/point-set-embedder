#pragma once

#include "honeybadger/Utils.cuh"

namespace honeybadger {
    // Implementation of remove vertex
    namespace remove_vertex {
        /**
         * Functions to update the objective value after removing a
         * vertex.
         */
        void objectiveValue(Solution::Data solution, int_t vertex, cudaStream_t stream = nullptr);

        /**
         * Functions to update the crossing number of each node
         * after removing a vertex.
         */
        // Spawn a thread for each node v.
        // Correcting the node crossings for every node v
        // by iterating incident edges of removed vertex and v.
        void nodeCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream = nullptr);

        /**
         * Functions to update the crossing number of each edge afer removing
         * an edge.
         */
        // Spawn a thread for each edge e.
        // Correcting the edge crossings for every edge e
        // by iterating incident edges of removed vertex and e.
        void edgeCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream = nullptr);

        /**
         * Functions to update the (m x m) matrix after removing a vertex.
         */
        void edgeCrossEdge(Solution::Data solution, int_t removed_vertex, cudaStream_t stream = nullptr);

        /**
         * Functions to update points' crossings after removing a vertex.
         */
        void pointCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream = nullptr);

        /**
         * Functions to update the crossing matrix (p x m)
         * after removing a vertex.
         */
        void pointCrossedEdges(Solution::Data solution, int_t removed_vertex, cudaStream_t stream = nullptr);
    }
}