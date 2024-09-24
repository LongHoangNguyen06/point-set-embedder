#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/private/operations/RemoveVertex.cuh"
#include "honeybadger/private/operations/RecomputeSolution.cuh"

namespace honeybadger {
    void Solution::remove(int_t vertex) {
        if (data.m_assignments[vertex] == NON_VALID_ASSIGNMENT) {
            throw_error("Trying to remove vertex " << vertex << ". But it is not assigned.",
                        std::invalid_argument)
        }
        int_t assigned_to = data.m_assignments[vertex];
        data.m_assignments[vertex] = NON_VALID_ASSIGNMENT;
        data.m_point_assigned_to[assigned_to] = NON_VALID_ASSIGNMENT;

        remove_vertex::objectiveValue(data, vertex);
        remove_vertex::nodeCrossings(data, vertex);
        remove_vertex::edgeCrossings(data, vertex);
        remove_vertex::edgeCrossEdge(data, vertex);

        cudaStream_t pointCrossingStream;
        gpu_error_check(cudaStreamCreate(&pointCrossingStream))
        remove_vertex::pointCrossings(data, vertex, pointCrossingStream);
        remove_vertex::pointCrossedEdges(data, vertex, pointCrossingStream);
        recompute_solution::isCorrect(data, pointCrossingStream);
        recompute_solution::isComplete(data, pointCrossingStream);
        cudaDeviceSynchronize();
        gpu_error_check(cudaStreamDestroy(pointCrossingStream))
    }
}


namespace honeybadger::remove_vertex {
    namespace kernel {
        __global__ void objectiveValue(Solution::Data solution, int_t vertex) {
            if (threadIdx.x == 0) {
                int_t objective_value = *solution.m_objective_value;
                for (int_t d = 0; d < solution.m_degree[vertex]; d++) {
                    int_t adjEdge = solution.m_incidence_matrix[vertex][d];
                    objective_value -= solution.m_edge_crossings[adjEdge];

                    Edge &edge = solution.m_edges[adjEdge]; // Reset m_node_crossings of neighbor nodes
                    solution.m_node_crossings[edge.opposite(vertex)] -= solution.m_edge_crossings[adjEdge];
                }
                *solution.m_objective_value = objective_value;
            }
        }
    }

    void objectiveValue(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        kernel::objectiveValue<<<1, 1, 0, stream>>>(solution, vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void nodeCrossings(Solution::Data solution, int_t removed_vertex) {
            solution.m_node_crossings[removed_vertex] = 0; // Reset node crossing of inserted removed_vertex

            int_t other_vertex = threadIdx.x + blockIdx.x * blockDim.x;
            if (other_vertex < solution.n && solution.isVertexVisible(other_vertex)) {

                // Iterate every incident edge of the removed vertex
                for (int_t d_removed_vertex = 0;
                     d_removed_vertex < solution.m_degree[removed_vertex]; d_removed_vertex++) {
                    int_t adj_edge_removed_vertex = solution.m_incidence_matrix[removed_vertex][d_removed_vertex];

                    // Iterate every incident edge of the other vertex
                    for (int_t d_other_vertex = 0;
                         d_other_vertex < solution.m_degree[other_vertex]; d_other_vertex++) {
                        int_t adj_edge_other_vertex = solution.m_incidence_matrix[other_vertex][d_other_vertex];

                        if (solution.m_edge_cross_edge[adj_edge_removed_vertex][adj_edge_other_vertex]) {
                            solution.m_node_crossings[other_vertex] -= 1; // Decrement
                        }

                    }
                }
            }
        }
    }

    void nodeCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream) {
        int_t required_threads = solution.n;
        int_t block_size = std::min(1024l, required_threads);
        int_t grid_size = divup(solution.n, block_size);
        kernel::nodeCrossings<<<grid_size, block_size, 0, stream>>>(solution, removed_vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void edgeCrossings(Solution::Data solution, int_t removed_vertex) {
            int_t other_edge = threadIdx.x + blockDim.x * blockIdx.x;
            if (other_edge < solution.m) {
                for (int_t d_removed_vertex = 0;
                     d_removed_vertex < solution.m_degree[removed_vertex]; d_removed_vertex++) {
                    int_t adj_edge_removed_vertex = solution.m_incidence_matrix[removed_vertex][d_removed_vertex];
                    solution.m_edge_crossings[adj_edge_removed_vertex] = 0;
                    solution.m_edge_crossings[other_edge] -= solution.m_edge_cross_edge[other_edge][adj_edge_removed_vertex];
                }
            }
        }
    }

    // Spawn a thread for each edge e.
    // Correcting the edge crossings for every edge e
    // by iterating incident edges of removed vertex and e.
    void edgeCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream) {
        int_t required_threads = solution.m;
        int_t block_size = std::min(1024l, required_threads);
        int_t grid_size = divup(solution.m, block_size);
        kernel::edgeCrossings<<<grid_size, block_size, 0, stream>>>(solution, removed_vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void edgeCrossEdge(Solution::Data solution, int_t removed_vertex) {
            int_t other_edge = threadIdx.x + blockIdx.x * blockDim.x;
            if (other_edge < solution.m) {
                for (int_t adj_edge_j = 0; adj_edge_j < solution.m_degree[removed_vertex]; adj_edge_j++) {
                    int_t adj_edge = solution.m_incidence_matrix[removed_vertex][adj_edge_j];
                    solution.m_edge_cross_edge[other_edge][adj_edge] = false;
                    solution.m_edge_cross_edge[adj_edge][other_edge] = false;
                }
            }
        }
    }

    void edgeCrossEdge(Solution::Data solution, int_t removed_vertex, cudaStream_t stream) {
        int_t required_threads = solution.m;
        int_t block_size = std::min(1024l, required_threads);
        int_t grid_size = divup(solution.m, block_size);
        kernel::edgeCrossEdge<<<grid_size, block_size, 0, stream>>>(solution, removed_vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void pointCrossings(Solution::Data solution, int_t removed_vertex) {
            int_t point_i = threadIdx.x + blockIdx.x * blockDim.x;
            if (point_i < solution.p) {
                for (int_t adj_edge_i = 0; adj_edge_i < solution.m_degree[removed_vertex]; adj_edge_i++) {
                    int_t adj_edge = solution.m_incidence_matrix[removed_vertex][adj_edge_i];
                    solution.m_point_crossings[point_i] -= solution.m_point_crossed_edge[point_i][adj_edge];
                }
            }
        }
    }

    void pointCrossings(Solution::Data solution, int_t removed_vertex, cudaStream_t stream) {
        int_t required_threads = solution.p;
        int_t block_size = std::min(1024l, required_threads);
        int_t grid_size = divup(solution.p, block_size);
        kernel::pointCrossings<<<grid_size, block_size, 0, stream>>>(solution, removed_vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void pointCrossedEdges(Solution::Data solution, int_t removed_vertex) {
            int_t point_i = threadIdx.x + blockIdx.x * blockDim.x;
            if (point_i < solution.p) {
                for (int_t adj_edge_i = 0; adj_edge_i < solution.m_degree[removed_vertex]; adj_edge_i++) {
                    int_t adj_edge = solution.m_incidence_matrix[removed_vertex][adj_edge_i];
                    solution.m_point_crossed_edge[point_i][adj_edge] = false;
                }
            }
        }
    }

    void pointCrossedEdges(Solution::Data solution, int_t removed_vertex, cudaStream_t stream) {
        int_t required_threads = solution.p;
        int_t block_size = std::min(1024l, required_threads);
        int_t grid_size = divup(solution.p, block_size);
        kernel::pointCrossedEdges<<<grid_size, block_size, 0, stream>>>(solution, removed_vertex);
        post_kernel_invocation_check
    }
}