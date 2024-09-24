#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/private/operations/RecomputeSolution.cuh"

namespace honeybadger {
    void batchAssign(Solution::Data solution, const std::vector<int_t> &assignments) {
        for (int_t i = 0; i < solution.n; i++) {
            solution.m_assignments[i] = NON_VALID_ASSIGNMENT;
        }
        for (int_t i = 0; i < solution.p; i++) {
            solution.m_point_assigned_to[i] = NON_VALID_ASSIGNMENT;
        }
        for (int_t i = 0; i < solution.n; i++) {
            solution.m_assignments[i] = assignments[i];
            if (assignments[i] != NON_VALID_ASSIGNMENT) {
                solution.m_point_assigned_to[assignments[i]] = i;
            }
        }
    }

    void batchAssign(Solution::Data solution, const std::vector<Point> &assignments) {
        std::vector<int_t> assignments_idx;
        for (const Point &point: assignments) {
            bool found = false;
            for (int_t point_idx = 0; point_idx < solution.p; point_idx++) {
                if (point == solution.m_points[point_idx]) {
                    assignments_idx.push_back(point_idx);
                    found = true;
                    break;
                }
            }
            if (!found) {
                assignments_idx.push_back(NON_VALID_ASSIGNMENT);
            }
        }
        batchAssign(solution, assignments_idx);
    }

    void Solution::update() {
        recompute_solution::edgeCrossEdge(this->data);
        recompute_solution::edgeCrossings(this->data);
        recompute_solution::objectiveValue(this->data);
        recompute_solution::nodeCrossings(this->data);

        cudaStream_t pointCrossingStream;
        gpu_error_check(cudaStreamCreate(&pointCrossingStream))
        recompute_solution::pointCrossedEdges(this->data, pointCrossingStream);
        recompute_solution::pointCrossings(this->data, pointCrossingStream);
        recompute_solution::isCorrect(this->data, pointCrossingStream);
        recompute_solution::isComplete(this->data, pointCrossingStream);
        cudaDeviceSynchronize();
        gpu_error_check(cudaStreamDestroy(pointCrossingStream))
    }

    void Solution::assign(const std::vector<Point> &points) {
        batchAssign(this->data, points);
        update();

        cudaDeviceSynchronize();
        if(!*this->data.m_is_correct) {
            throw_error("Input assignment is not correct", std::invalid_argument);
        }
    }
}

namespace honeybadger::recompute_solution {
    namespace kernel {
        [[maybe_unused]] __global__ void edgeCrossEdge(Solution::Data solution) {
            int_t x = threadIdx.x + blockIdx.x * blockDim.x;
            int_t y = threadIdx.y + blockIdx.y * blockDim.y;
            if (x < solution.m && y < solution.m) {
                solution.m_edge_cross_edge[y][x] = solution.doesEdgeCrossesEdge(y, x);
            }
        }
    }

    void edgeCrossEdge(Solution::Data solution, cudaStream_t stream) {
        constexpr int_t size = 32;
        kernel::edgeCrossEdge<<<dim3(divup(solution.m, size), divup(solution.m, size)), dim3(size,
                                                                                             size), 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void edgeCrossings(Solution::Data solution) {
            int_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < solution.m) {
                solution.m_edge_crossings[i] = 0;
                for (int_t j = 0; j < solution.m; j++) {
                    solution.m_edge_crossings[i] += solution.m_edge_cross_edge[i][j];
                }
            }
        }
    }

    // Launch m threads.
    void edgeCrossings(Solution::Data solution, cudaStream_t stream) {
        int_t threads_per_block = std::min(1024l, solution.m);
        kernel::edgeCrossings<<<divup(solution.m, threads_per_block), threads_per_block, 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        [[maybe_unused]] __global__ void objectiveValue(Solution::Data solution) {
            if (threadIdx.x == 0) {
                *solution.m_objective_value = 0; // Set default value
                for (int_t x = 0; x < solution.m; x++) {
                    *solution.m_objective_value += solution.m_edge_crossings[x];
                }
                *solution.m_objective_value /= 2;
            }
        }
    }

    // Launch 1 thread.
    void objectiveValue(Solution::Data solution, cudaStream_t stream) {
        kernel::objectiveValue<<<1, 1, 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void nodeCrossings(Solution::Data solution) {
            int_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < solution.n) {
                solution.m_node_crossings[i] = 0; // Set default value
                for (int_t j = 0; j < solution.m_degree[i]; j++) {
                    auto &incident_edge_j = solution.m_incidence_matrix[i][j];
                    solution.m_node_crossings[i] += solution.m_edge_crossings[incident_edge_j];
                }
            }
        }
    }

    // Launch n threads.
    void nodeCrossings(Solution::Data solution, cudaStream_t stream) {
        int_t threads_per_block = std::min(1024l, solution.n);
        kernel::nodeCrossings<<<divup(solution.n, threads_per_block), threads_per_block, 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void pointCrossedEdges(Solution::Data solution) {
            int_t x = threadIdx.x + blockIdx.x * blockDim.x; // edge's index
            int_t y = threadIdx.y + blockIdx.y * blockDim.y; // point's index
            if (x < solution.m && y < solution.p) {
                solution.m_point_crossed_edge[y][x] = solution.isPointCrossedByEdge(y, x);
            }
        }
    }

    // Launch (p x m) threads.
    void pointCrossedEdges(Solution::Data solution, cudaStream_t stream) {
        constexpr int_t size = 32;
        kernel::pointCrossedEdges<<<dim3(divup(solution.m, size), divup(solution.p, size)), dim3(size,
                                                                                                 size), 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        __global__ void pointCrossings(Solution::Data solution) {
            extern __shared__ int_t crossings[];
            crossings[threadIdx.x] = 0;
            for (int_t j = threadIdx.x; j < solution.m; j += blockDim.x) {
                crossings[threadIdx.x] += solution.m_point_crossed_edge[blockIdx.x][j];
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                solution.m_point_crossings[blockIdx.x] = 0;
                for (int_t i = 0; i < blockDim.x; i++) {
                    solution.m_point_crossings[blockIdx.x] += crossings[i];
                }
            }
        }
    }

    // Launch p blocks, each block for a point.
    void pointCrossings(Solution::Data solution, cudaStream_t stream) {
        int_t threads_per_block = 32;
        kernel::pointCrossings<<<solution.p,
                threads_per_block,
                threads_per_block *
                sizeof(solution.m_point_crossings[0]),
                stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        [[maybe_unused]] __global__ void isCorrectC1(Solution::Data solution) {
            int_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
            if (vertex < solution.n) {
                // If vertex is not assigned, no need to check
                if (solution.m_assignments[vertex] == NON_VALID_ASSIGNMENT) {
                    return;
                }
                // Check C1
                if (solution.m_assignments[vertex] < 0 || solution.m_assignments[vertex] >= solution.p) {
                    *solution.m_is_correct = false;
                    return;
                }
                // Check C1.1
                if (solution.m_point_assigned_to[solution.m_assignments[vertex]] != vertex) {
                    *solution.m_is_correct = false;
                    return;
                }
            }
        }

        [[maybe_unused]] __global__ void isCorrectC2AndC3(Solution::Data solution) {
            int_t point_idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (point_idx < solution.p) {
                // If point is not assigned, not need to check
                if (solution.m_point_assigned_to[point_idx] == NON_VALID_ASSIGNMENT) {
                    return;
                }
                // Check C2
                if (solution.m_point_assigned_to[point_idx] < 0 || solution.m_point_assigned_to[point_idx] >= solution.n) {
                    *solution.m_is_correct = false;
                    return;
                }
                if (solution.m_assignments[solution.m_point_assigned_to[point_idx]] != point_idx) {
                    *solution.m_is_correct = false;
                    return;
                }
                // Check C3
                if (solution.m_point_crossings[point_idx] > 0) {
                    *solution.m_is_correct = false;
                    return;
                }
            }
        }

        [[maybe_unused]] __global__ void isCorrectC4(Solution::Data solution) {
            int_t vertex1 = threadIdx.x + blockIdx.x * blockDim.x;
            int_t vertex2 = threadIdx.y + blockIdx.y * blockDim.y;
            if (vertex1 < solution.n && vertex2 < solution.n) {
                // If both vertices are same, no need to check
                if (vertex1 == vertex2) {
                    return;
                }
                // If one of both vertices are not assigned, no need to check
                if (solution.m_assignments[vertex1] == NON_VALID_ASSIGNMENT ||
                        solution.m_assignments[vertex2] == NON_VALID_ASSIGNMENT) {
                    return;
                }
                // Check C4
                if (solution.m_assignments[vertex1] == solution.m_assignments[vertex2]) {
                    *solution.m_is_correct = false;
                    return;
                }
            }
        }
    }

    void isCorrect(Solution::Data solution, cudaStream_t stream) {
        // Check C1
        cudaDeviceSynchronize();
        *solution.m_is_correct = true; // Set default value

        cudaDeviceSynchronize();
        int_t threads_per_block = std::min(1024l, solution.n);
        kernel::isCorrectC1<<<divup(solution.n, threads_per_block), threads_per_block, 0, stream>>>(solution);
        post_kernel_invocation_check

        // Check C2 and C3
        cudaDeviceSynchronize();
        if(!*solution.m_is_correct) {
            return;
        }
        threads_per_block = std::min(1024l, solution.p);
        kernel::isCorrectC2AndC3<<<divup(solution.p, threads_per_block), threads_per_block, 0, stream>>>(solution);
        post_kernel_invocation_check

        // Check C4
        cudaDeviceSynchronize();
        if(!*solution.m_is_correct) {
            return;
        }
        auto size = 32;
        kernel::isCorrectC4<<<dim3(divup(solution.n, size), divup(solution.n, size)), dim3(size, size), 0, stream>>>(solution);
        post_kernel_invocation_check
    }

    namespace kernel {
        [[maybe_unused]] __global__ void isComplete(Solution::Data solution) {
            int_t vertex = threadIdx.x + blockIdx.x * blockDim.x;
            if (vertex < solution.n) {
                if (solution.m_assignments[vertex] < 0 || solution.m_assignments[vertex] >= solution.p) {
                    *solution.m_is_complete = false;
                }
            }
        }
    }

    // Launch p thread.
    void isComplete(Solution::Data solution, cudaStream_t stream) {
        cudaDeviceSynchronize();
        *solution.m_is_complete = true; // Set default value
        cudaDeviceSynchronize();
        int_t threads_per_block = std::min(1024l, solution.n);
        kernel::isComplete<<<divup(solution.n, threads_per_block), threads_per_block, 0, stream>>>(solution);
        post_kernel_invocation_check
    }
}