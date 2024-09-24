#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/Geometry.cuh"
#include "honeybadger/Utils.cuh"
#include "honeybadger/private/operations/InsertVertex.cuh"
#include "honeybadger/private/operations/RecomputeSolution.cuh"

namespace honeybadger {
    void Solution::insert(int_t vertex, int_t point_idx) {
        if (data.m_assignments[vertex] != NON_VALID_ASSIGNMENT) {
            throw_error("Trying to insert vertex " << vertex << " into point " << point_idx
                                                   << " vertex is already assigned to " <<
                                                   data.m_assignments[vertex] << std::endl,
                        std::invalid_argument)
        }
        if (data.m_point_assigned_to[point_idx] != NON_VALID_ASSIGNMENT) {
            throw_error("Trying to insert vertex " << vertex << " into point " << point_idx
                                                   << " point is already assigned to " <<
                                                   data.m_point_assigned_to[point_idx] << std::endl,
                        std::invalid_argument)
        }
        data.m_assignments[vertex] = point_idx;
        data.m_point_assigned_to[point_idx] = vertex;

        operations::insert_vertex::edgeCrossEdge(this->data, vertex);
        operations::insert_vertex::edgeCrossings(this->data, vertex);
        operations::insert_vertex::objectiveValue(this->data, vertex);
        recompute_solution::nodeCrossings(this->data);

        cudaStream_t pointCrossingStream;
        gpu_error_check(cudaStreamCreate(&pointCrossingStream))
        operations::insert_vertex::pointCrossedEdges(this->data, vertex, pointCrossingStream);
        recompute_solution::pointCrossings(this->data, pointCrossingStream);
        recompute_solution::isCorrect(this->data, pointCrossingStream);
        recompute_solution::isComplete(this->data, pointCrossingStream);
        cudaDeviceSynchronize();
        gpu_error_check(cudaStreamDestroy(pointCrossingStream))
    }
}

namespace honeybadger::operations::insert_vertex {
    namespace kernel {
        __global__ void edgeCrossEdge(Solution::Data solution, int_t vertex) {
            int_t edge_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (edge_x_idx < solution.m) {
                for (int_t y = 0; y < solution.m_degree[vertex]; y++) {
                    int_t edge_y_idx = solution.m_incidence_matrix[vertex][y];
                    bool cross = solution.doesEdgeCrossesEdge(edge_x_idx, edge_y_idx);
                    solution.m_edge_cross_edge[edge_y_idx][edge_x_idx] = cross;
                    solution.m_edge_cross_edge[edge_x_idx][edge_y_idx] = cross;
                    solution.m_edge_crossings[edge_x_idx] += cross;
                }
            }
        }
    }

    void edgeCrossEdge(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        int_t block_size = std::min(1024l, solution.m);
        int_t grid_size = divup(solution.m, block_size);
        kernel::edgeCrossEdge<<<grid_size, block_size, 0, stream>>>(solution, vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void edgeCrossings(Solution::Data solution, int_t vertex) {
            int_t incident_edge_i = threadIdx.x + blockIdx.x * blockDim.x;
            if (incident_edge_i < solution.m_degree[vertex]) {
                int_t incident_edge_idx = solution.m_incidence_matrix[vertex][incident_edge_i];
                solution.m_edge_crossings[incident_edge_idx] = 0;  // Set default value
                for (int_t visible_edge_i = 0; visible_edge_i < solution.m; visible_edge_i++) {

                    solution.m_edge_crossings[incident_edge_idx] += solution.m_edge_cross_edge[incident_edge_idx][visible_edge_i];
                }
            }
        }
    }

    // This only update the edge crossings of incident edges of the vertex
    void edgeCrossings(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        int_t required_threads = solution.m_degree[vertex];
        if (required_threads > 0) {
            int_t block_size = std::min(1024l, required_threads);
            int_t grid_size = divup(required_threads, block_size);
            kernel::edgeCrossings<<<grid_size, block_size, 0, stream>>>(solution, vertex);
            post_kernel_invocation_check
        }
    }

    namespace kernel {
        __global__ void objectiveValue(Solution::Data solution, int_t vertex) {
            if (threadIdx.x == 0) {
                int_t additional_crossings = 0;
                for (int_t d = 0; d < solution.m_degree[vertex]; d++) {
                    int_t adjEdge = solution.m_incidence_matrix[vertex][d];
                    additional_crossings += solution.m_edge_crossings[adjEdge];
                }
                *solution.m_objective_value += additional_crossings;
            }
        }
    }

    void objectiveValue(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        kernel::objectiveValue<<<1, 1, 0, stream>>>(solution, vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void pointCrossedEdges(Solution::Data solution, int_t vertex) {
            int_t point_y_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (point_y_idx < solution.p) {
                for (int_t x = 0; x < solution.m_degree[vertex]; x++) {
                    int_t edge_y_idx = solution.m_incidence_matrix[vertex][x];
                    solution.m_point_crossed_edge[point_y_idx][edge_y_idx] = solution.isPointCrossedByEdge(
                            point_y_idx,
                            edge_y_idx);
                }
            }
        }
    }

    void pointCrossedEdges(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        int_t block_size = std::min(1024l, solution.p);
        int_t grid_size = divup(solution.p, block_size);
        kernel::pointCrossedEdges<<<grid_size, block_size, 0, stream>>>(solution, vertex);
        post_kernel_invocation_check
    }
}