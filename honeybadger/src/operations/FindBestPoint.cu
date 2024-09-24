#include "honeybadger/private/operations/FindBestPoint.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/Geometry.cuh"
#include "honeybadger/Utils.cuh"
#include <thrust/extrema.h>

namespace honeybadger {
std::vector<int_t> Solution::findBestPoint(int_t vertex) {
        if (data.m_assignments[vertex] != NON_VALID_ASSIGNMENT) {
            throw_error("Trying to find a node for vertex "
                        << vertex << " but vertex is already assigned to point "
                        << data.m_assignments[vertex] << std::endl,
                        std::invalid_argument)
        }
        find_best_points::computeNumAvailableAndAssignedPoints(data);
        find_best_points::computeAvailablePointsCrossAssignedPoints(data, vertex);
        find_best_points::computeNumPossiblePoints(data);
        find_best_points::computePossiblePointsCrossingNumber(data, vertex);
        find_best_points::computeNumBestPoints(data);
        cudaDeviceSynchronize();
        return {data.m_best_points_indices, data.m_best_points_indices + *data.m_num_best_points};
    }
}

namespace honeybadger::find_best_points {
    namespace kernel {
        __global__ void computeNumAvailableAndAssignedPoints(Solution::Data solution) {
            if (threadIdx.x == 0) {
                int_t available_points = 0;
                int_t assigned_points = 0;

                for (int_t i = 0; i < solution.p; i++) {
                    bool assigned = solution.isPointAssigned(i);
                    bool crossed = solution.isPointCrossed(i);
                    if (!assigned && !crossed) {
                        solution.m_available_points_indices[available_points++] = i;
                    } else if (!crossed) {
                        solution.m_assigned_points_indices[assigned_points++] = i;
                    }
                }

                *solution.m_num_assigned_points = assigned_points;
                *solution.m_num_available_points = available_points;
            }
        }
    }

    // Spawn a single thread.
    void computeNumAvailableAndAssignedPoints(Solution::Data solution, cudaStream_t stream) {
        kernel::computeNumAvailableAndAssignedPoints<<<1, 1, 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void computeAvailablePointsCrossAssignedPoints(Solution::Data solution, int_t vertex) {
            int_t i = blockIdx.x; // Position of thread's available point
            solution.m_available_points_cross_assigned_points[i] = false; // Set default value
            __syncthreads();

            int_t available_point_idx = solution.m_available_points_indices[i]; // Global position of available point
            Point available_point = solution.m_points[available_point_idx]; // Available point

            for (int_t d = 0; d < solution.m_degree[vertex]; d++) { // Iterate every visible incident edge

                int_t adj_edge_idx = solution.m_incidence_matrix[vertex][d]; // Fetch incident edge and neighbor node
                auto &adj_edge = solution.m_edges[adj_edge_idx];
                auto neighbor = adj_edge.opposite(vertex);

                if (solution.isVertexVisible(neighbor)) {
                    int_t neighbor_assignment = solution.m_assignments[neighbor];
                    Point neighbor_point = solution.m_points[neighbor_assignment];
                    Segment hypothetical_segment{available_point, neighbor_point};

                    for (int_t k = threadIdx.x; k < solution.p; k += blockDim.x) { // Iterate every point
                        if (solution.isPointAssigned(k) &&
                            geometry::cross(hypothetical_segment, solution.m_points[k])) {
                            solution.m_available_points_cross_assigned_points[i] = true;
                            break;
                        }
                    }

                }
            }
        }
    }

    void computeAvailablePointsCrossAssignedPoints(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        cudaDeviceSynchronize();
        if (*solution.m_num_available_points == 0) {
            return;
        }
        kernel::computeAvailablePointsCrossAssignedPoints<<<*solution.m_num_available_points, 128, 0, stream>>>(solution, vertex);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void computeNumPossiblePoints(Solution::Data solution) {
            if (threadIdx.x == 0) {
                int_t num_possible_points = 0;
                for (int_t i = 0; i < *solution.m_num_available_points; i++) {
                    if (!solution.m_available_points_cross_assigned_points[i]) {
                        solution.m_possible_points_indices[num_possible_points++] = solution.m_available_points_indices[i];
                    }
                }
                *solution.m_num_possible_points = num_possible_points;
            }
        }
    }

    // Spawn a single thread.
    void computeNumPossiblePoints(Solution::Data solution, cudaStream_t stream) {
        kernel::computeNumPossiblePoints<<<1, 1, 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void
        computePossiblePointsCrossingNumber(Solution::Data solution, int_t vertex, int_t bytes_for_crossings,
                                            int_t bytes_for_incident_segments) {
            extern __shared__ bool buffer[];
            auto *crossings = (int_t *) &buffer[0];
            auto *incident_segments = (Segment *) &buffer[bytes_for_crossings];
            auto *is_incident_segments_visible = &buffer[bytes_for_crossings + bytes_for_incident_segments];

            // Initialize thread's data
            int64_t i = blockIdx.x;
            crossings[threadIdx.x] = 0;
            int_t possible_point_idx = solution.m_possible_points_indices[i]; // Global position of possible point
            Point possible_point = solution.m_points[possible_point_idx]; // Possible point

            // Load incident segments into shared memory
            if (threadIdx.x == 0) {
                for (int_t d = 0; d < solution.m_degree[vertex]; d++) { // Iterate every visible incident edge
                    int_t adj_edge_idx = solution.m_incidence_matrix[vertex][d];
                    auto &adj_edge = solution.m_edges[adj_edge_idx];
                    auto neighbor = adj_edge.opposite(vertex);
                    is_incident_segments_visible[d] = solution.isVertexVisible(neighbor);
                    if (is_incident_segments_visible[d]) {
                        int_t neighbor_assignment = solution.m_assignments[neighbor];
                        Point neighbor_point = solution.m_points[neighbor_assignment];
                        incident_segments[d] = Segment{possible_point, neighbor_point};
                    }
                }
            }
            __syncthreads();

            // Iterate every edge and check for crossings
            for (int_t m = threadIdx.x; m < solution.m; m += blockDim.x) {
                if (solution.isEdgeVisible(m)) {
                    Segment segment_m = solution.getVisibleSegment(m);
                    for (int_t d = 0; d < solution.m_degree[vertex]; d++) {
                        if (is_incident_segments_visible[d]) {
                            crossings[threadIdx.x] += geometry::intersect(incident_segments[d], segment_m);
                        }
                    }
                }
            }

            __syncthreads();

            // Sum up
            if (threadIdx.x == 0) {
                solution.m_possible_points_crossing_number[i] = 0;
                for (int_t t = 0; t < blockDim.x; t++) {
                    solution.m_possible_points_crossing_number[i] += crossings[t];
                }
            }
        }
    }

    // Spawn a thread block for every possible point.
    void computePossiblePointsCrossingNumber(Solution::Data solution, int_t vertex, cudaStream_t stream) {
        cudaDeviceSynchronize();
        if (*solution.m_num_possible_points == 0) {
            return;
        }
        int_t threads_per_block = 256;

        // Compute size of shared memory
        auto bytes_for_crossings = threads_per_block *
                                   sizeof(solution.m_possible_points_crossing_number[0]);
        auto bytes_for_incident_segments = sizeof(Segment) * solution.m_degree[vertex];
        auto bytes_for_is_segment_visible = sizeof(bool) * solution.m_degree[vertex];
        auto total_bytes = bytes_for_crossings + bytes_for_incident_segments + bytes_for_is_segment_visible;

        // Launch kernels
        kernel::computePossiblePointsCrossingNumber<<<*
                solution.m_num_possible_points,
                threads_per_block,
                total_bytes, stream>>>(solution, vertex, bytes_for_crossings, bytes_for_incident_segments);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void computeNumBestPoints(Solution::Data solution) {
            if (threadIdx.x == 0) {
                auto min_crossing = *thrust::min_element(thrust::seq,
                                                         solution.m_possible_points_crossing_number,
                                                         solution.m_possible_points_crossing_number +
                                                         *solution.m_num_possible_points);

                int_t num_best_points = 0;
                for (int_t i = 0; i < *solution.m_num_possible_points; i++) {
                    if (solution.m_possible_points_crossing_number[i] == min_crossing) {
                        solution.m_best_points_indices[num_best_points++] = solution.m_possible_points_indices[i];
                    }
                }
                *solution.m_num_best_points = num_best_points;
            }
        }
    }

    // Spawn a single thread.
    void computeNumBestPoints(Solution::Data solution, cudaStream_t stream) {
        kernel::computeNumBestPoints<<<1, 1, 0, stream>>>(solution);
        post_kernel_invocation_check
    }
}