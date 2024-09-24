#include "honeybadger/datastructure/Graph.cuh"
#include "honeybadger/Utils.cuh"
#include <vector>
#include <algorithm>

namespace honeybadger {
    std::vector<std::vector<int_t>> getIncidenceMatrix(const std::vector<Edge> &edges, int_t n) {
        std::vector<std::vector<int_t>> incidence_matrix(n);

        int_t i = 0;
        for (const auto &edge: edges) {
            incidence_matrix[edge.m_source].push_back(i);
            incidence_matrix[edge.m_target].push_back(i);
            i++;
        }
        return incidence_matrix;
    }

    Graph::Graph(const std::vector<Edge> &edges, int_t n) {
        if (hasDuplicate(edges)) {
            throw_error("Input edges are not simple. Duplicate edges found" << std::endl, std::invalid_argument)
        }
        assert(!hasDuplicate(edges));
        this->n = n;
        this->m = static_cast<int_t>(edges.size());

        // Create and sort edges
        for (const auto &edge: edges) {
            if (!edge.is_valid(n)) {
                throw_error("Invalid edge found " << edge << "with this has " << this->n << " nodes " << std::endl,
                            std::invalid_argument)
            }
        }

        // m_edges
        gpu_error_check(robust_cuda_malloc_managed(this->m_edges, this->m))
        this->edges([&](auto i) {
            this->m_edges[i] = edges[i];
        });

        // m_incidence_matrix
        auto incidence_matrix = getIncidenceMatrix(edges, n);
        gpu_error_check(robust_cuda_malloc_managed(this->m_incidence_matrix, n))

        // m_degree
        gpu_error_check(robust_cuda_malloc_managed(this->m_degree, n))

        // Transfer CPU data to GPU
        for (size_t i = 0; i < incidence_matrix.size(); i++) {
            const auto &row = incidence_matrix[i];
            auto d = static_cast<int_t>(row.size());
            this->m_degree[i] = d;

            // m_incidence_matrix
            gpu_error_check(robust_cuda_malloc_managed(this->m_incidence_matrix[i], d))

            // m_adjacency_list
            for (size_t j = 0; j < row.size(); j++) {
                this->m_incidence_matrix[i][j] = row[j];
            }
        }
    }

    Graph::~Graph() {
        gpu_error_check(cudaFree(this->m_edges))
        gpu_error_check(cudaFree(this->m_degree))
        vertices([&](auto i) {
            gpu_error_check(cudaFree(this->m_incidence_matrix[i]))
        });
        gpu_error_check(cudaFree(this->m_incidence_matrix))
    }

    std::ostream &operator<<(std::ostream &os, const Edge &edge) {
        os << "(" << edge.m_source << " -> " << edge.m_target << ")";
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const Graph &graph) {
        os << "Graph{";
        os << std::endl << "\tEdges: ";
        for (int_t i = 0; i < graph.m; i++) {
            os << std::endl << "\t" << i << ": " << graph.m_edges[i];
        }
        os << std::endl << "\tIncidence matrix:";
        for (int_t i = 0; i < graph.n; i++) {
            os << std::endl << "\t\t" << i << ": ";
            for (int_t j = 0; j < graph.m_degree[i]; j++) {
                os << graph.m_incidence_matrix[i][j] << " ";
            }
        }
        os << "}";
        return os;
    }

    bool Graph::operator==(const Graph &rhs) const {
        if (n != rhs.n) {
            LOGGING << "n != n";
            return false;
        }
        if (m != rhs.m) {
            LOGGING << "m != m";
            return false;
        }
        for (size_t i = 0; i < m; i++) {
            if (m_edges[i] != rhs.m_edges[i]) {
                LOGGING << "Edge not equal" << std::endl;
                return false;
            }
        }
        for (size_t i = 0; i < n; i++) {
            if (m_degree[i] != rhs.m_degree[i]) {
                LOGGING << "Degree not equal" << std::endl;
                return false;
            }
        }
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < m_degree[i]; d++) {
                if (m_incidence_matrix[i][d] != rhs.m_incidence_matrix[i][d]) {
                    LOGGING << "Incidence not equal" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    bool Graph::operator!=(const Graph &rhs) const {
        return !(rhs == *this);
    }
}