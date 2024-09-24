#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/Geometry.cuh"
#include "honeybadger/Utils.cuh"
#include <cmath>
#include <algorithm>

namespace honeybadger {
    Solution::Data::Data(Problem::Ptr problem) {
        n = problem->m_graph->n;
        m = problem->m_graph->m;
        p = problem->p;
        m_id = problem->m_id;

        m_graph = problem->m_graph.get();
        m_problem = problem.get();
        m_edges = problem->m_graph->m_edges;
        m_incidence_matrix = problem->m_graph->m_incidence_matrix;
        m_degree = problem->m_graph->m_degree;
        m_points = problem->m_points;

        gpu_error_check(robust_cuda_malloc_managed(m_assignments, n))
        for (size_t i = 0; i < n; i++) {
            m_assignments[i] = NON_VALID_ASSIGNMENT;
        }

        gpu_error_check(robust_cuda_malloc_managed(m_point_assigned_to, p))
        for (size_t i = 0; i < p; i++) {
            m_point_assigned_to[i] = NON_VALID_ASSIGNMENT;
        }

        gpu_error_check(robust_cuda_malloc_managed(m_edge_cross_edge, m))
        for (size_t i = 0; i < m; i++) {
            gpu_error_check(robust_cuda_malloc_managed(m_edge_cross_edge[i], m))
        }
        for (size_t j = 0; j < m; j++) {
            for (size_t k = 0; k < m; k++) {
                m_edge_cross_edge[j][k] = false;
            }
        }

        gpu_error_check(robust_cuda_malloc_managed(m_edge_crossings, m))

        gpu_error_check(robust_cuda_malloc_managed(m_objective_value, 1))
        *m_objective_value = 0;

        gpu_error_check(robust_cuda_malloc_managed(m_node_crossings, n))

        gpu_error_check(robust_cuda_malloc_managed(m_point_crossed_edge, p))
        for (size_t i = 0; i < p; i++) {
            gpu_error_check(robust_cuda_malloc_managed(m_point_crossed_edge[i], m))
        }

        gpu_error_check(robust_cuda_malloc_managed(m_point_crossings, p))

        gpu_error_check(robust_cuda_malloc_managed(m_is_correct, 1))
        *m_is_correct = true;
        gpu_error_check(robust_cuda_malloc_managed(m_is_complete, 1))
        *m_is_complete = false;

        gpu_error_check(robust_cuda_malloc_managed(m_available_points_indices, p))

        gpu_error_check(robust_cuda_malloc_managed(m_num_available_points, 1))
        *m_num_available_points = 0;

        gpu_error_check(robust_cuda_malloc_managed(m_assigned_points_indices, p))

        gpu_error_check(robust_cuda_malloc_managed(m_num_assigned_points, 1))
        *m_num_assigned_points = 0;

        gpu_error_check(
                robust_cuda_malloc_managed(m_available_points_cross_assigned_points, p))

        gpu_error_check(robust_cuda_malloc_managed(m_possible_points_indices, p))

        gpu_error_check(robust_cuda_malloc_managed(m_num_possible_points, 1))
        *m_num_possible_points = 0;

        gpu_error_check(robust_cuda_malloc_managed(m_possible_points_crossing_number, p))

        gpu_error_check(robust_cuda_malloc_managed(m_best_points_indices, p))

        gpu_error_check(robust_cuda_malloc_managed(m_num_best_points, 1))
        *m_num_best_points = 0;

    }

    Solution::Solution(Problem::Ptr problem) : m_graph(problem->m_graph), m_problem(problem), m_own_data(true),
                                               data(Data(problem)) {
    }

    Solution::Solution(const Solution &solution) : data(solution.data), m_own_data(false){
    }

    Solution::~Solution() {
        if (m_own_data) {
            gpu_error_check(cudaFree(this->data.m_assignments))

            gpu_error_check(cudaFree(this->data.m_point_assigned_to))

            for (size_t i = 0; i < this->data.m; i++) {
                gpu_error_check(cudaFree(this->data.m_edge_cross_edge[i]))
            }
            gpu_error_check(cudaFree(this->data.m_edge_cross_edge))

            gpu_error_check(cudaFree(this->data.m_edge_crossings))

            gpu_error_check(cudaFree(this->data.m_objective_value))

            gpu_error_check(cudaFree(this->data.m_node_crossings))

            for (size_t i = 0; i < this->data.p; i++) {
                gpu_error_check(cudaFree(this->data.m_point_crossed_edge[i]))
            }
            gpu_error_check(cudaFree(this->data.m_point_crossed_edge))

            gpu_error_check(cudaFree(this->data.m_point_crossings))

            gpu_error_check(cudaFree(this->data.m_is_correct))

            gpu_error_check(cudaFree(this->data.m_is_complete))

            gpu_error_check(cudaFree(this->data.m_available_points_indices))

            gpu_error_check(cudaFree(this->data.m_num_available_points))

            gpu_error_check(cudaFree(this->data.m_assigned_points_indices))

            gpu_error_check(cudaFree(this->data.m_num_assigned_points))

            gpu_error_check(cudaFree(this->data.m_available_points_cross_assigned_points))

            gpu_error_check(cudaFree(this->data.m_possible_points_indices))

            gpu_error_check(cudaFree(this->data.m_num_possible_points))

            gpu_error_check(cudaFree(this->data.m_possible_points_crossing_number))

            gpu_error_check(cudaFree(this->data.m_best_points_indices))

            gpu_error_check(cudaFree(this->data.m_num_best_points))
        }
    }

    bool Solution::operator==(const Solution &rhs) const {
        return this->data == rhs.data;
    }

    bool Solution::operator!=(const Solution &rhs) const {
        return !(*this == rhs);
    }

    std::ostream &Solution::operator<<(std::ostream &os) {
        //os << this->data;
        return os;
    }

    bool Solution::Data::operator==(const Solution::Data &rhs) const {
        const auto eq1_ = [](auto *ptr1, auto *ptr2, size_t cols) -> bool {
            return std::equal(ptr1, ptr1 + cols, ptr2);
        };
        const auto eq2_ = [eq1_](auto **ptr1, auto **ptr2, size_t rows, size_t cols) -> bool {
            return std::equal(ptr1, ptr1 + rows, ptr2, [eq1_, cols](auto a, auto b) {
                return eq1_(a, b, cols);
            });
        };
        const auto eq3_ = [](auto *ptr1, auto *ptr2, size_t cols) -> bool {
            return std::all_of(ptr1, ptr1 + cols, [&](auto in) -> bool {
                return std::find(ptr2, ptr2 + cols, in) != ptr2 + cols;
            });
        };

#define eq1(ptr, n) {if(!eq1_(ptr, rhs.ptr, n)) {LOGGING << "Not equal " << #ptr << std::endl; return false;}}
#define eq2(ptr, rows, cols) {if(!eq2_(ptr, rhs.ptr, rows, cols)) {LOGGING << "Not equal " << #ptr << std::endl; return false;}}
#define eq3(ptr, n) {if(!eq3_(ptr, rhs.ptr, n)) {LOGGING << "Not equal " << #ptr << std::endl; return false;}}

        eq1(m_assignments, n)
        eq1(m_point_assigned_to, p)

        eq2(m_edge_cross_edge, m, m)
        eq1(m_edge_crossings, m)
        eq1(m_objective_value, 1)
        eq1(m_node_crossings, n)
        eq2(m_point_crossed_edge, p, m)
        eq1(m_point_crossings, p)
        eq1(m_is_correct, 1)
        eq1(m_is_complete, 1)

        eq1(m_num_available_points, 1)
        eq3(m_available_points_indices, *m_num_available_points)
        eq1(m_num_assigned_points, 1)
        eq3(m_assigned_points_indices, *m_num_assigned_points)

        eq1(m_available_points_cross_assigned_points, p)
        eq1(m_num_possible_points, 1)
        eq3(m_possible_points_indices, *m_num_possible_points)

        eq1(m_possible_points_crossing_number, p)
        eq1(m_num_best_points, 1)
        eq3(m_best_points_indices, *m_num_best_points)

        if (*m_graph != *rhs.m_graph) {
            LOGGING << "Graph not equal" << std::endl;
            return false;
        }

        if (*m_problem != *rhs.m_problem) {
            LOGGING << "Problem not equal" << std::endl;
            return false;
        }

        if (n != rhs.n) {
            LOGGING << "n not equal" << std::endl;
            return false;
        }

        if (m != rhs.m) {
            LOGGING << "m not equal" << std::endl;
            return false;
        }

        if (p != rhs.p) {
            LOGGING << "p not equal" << std::endl;
            return false;
        }

        return true;
    }

    bool Solution::Data::operator!=(const Solution::Data &rhs) const {
        return !(*this == rhs);
    }

    std::ostream &Solution::Data::operator<<(std::ostream &os) {
        const auto print1 = [&](auto *ptr, size_t cols, auto str) {
            os << std::endl << "\t" << str << ": ";
            for (int_t i = 0; i < cols; i++) {
                os << ptr[i] << " ";
            }
        };
        const auto print2 = [&](auto **ptr, size_t rows, size_t cols, auto str) {
            os << std::endl << "\t" << str << ": ";
            for (int_t i = 0; i < rows; i++) {
                os << std::endl << "\t\t";
                for (int_t j = 0; j < cols; j++) {
                    os << ptr[i][j] << " ";
                }
            }
        };

#define p1(ptr, cols) {print1(this->ptr, cols, #ptr);}
#define p2(ptr, cols, rows) {print2(this->ptr, cols, rows, #ptr);}

        os << "Solution{";
        p1(m_assignments, this->n)
        p1(m_point_assigned_to, this->p)

        p2(m_edge_cross_edge, this->m, this->m)
        p1(m_edge_crossings, this->m)
        p1(m_objective_value, 1)
        p1(m_node_crossings, this->n)
        p2(m_point_crossed_edge, this->p, this->m)
        p1(m_point_crossings, this->p)
        p1(m_is_correct, 1)
        p1(m_is_complete, 1)

        p1(m_available_points_indices, this->p)
        p1(m_num_available_points, 1)
        p1(m_assigned_points_indices, this->p)
        p1(m_num_assigned_points, 1)

        p1(m_available_points_cross_assigned_points, this->p)
        p1(m_possible_points_indices, this->p)
        p1(m_num_possible_points, 1)

        p1(m_possible_points_crossing_number, this->p)
        p1(m_best_points_indices, this->p)
        p1(m_num_best_points, 1)
        os << std::endl << "}";

        return os;
    }
}
