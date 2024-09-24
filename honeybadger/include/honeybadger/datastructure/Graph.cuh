#pragma once

#include <cinttypes>
#include <vector>
#include <ostream>
#include <cmath>
#include <set>
#include <memory>
#include <thrust/universal_vector.h>
#include "honeybadger/Utils.cuh"

namespace honeybadger {

    /**
     * Edge presentation.
     * From = smaller node.
     * To = larger node.
     */
    typedef struct Edge {
        int_t m_source;
        int_t m_target;

        inline __host__ __device__ Edge(int_t source, int_t target): m_source(source < target ? source : target),
                                                                     m_target(source < target ? target : source)  {

        }

        inline __host__ __device__ bool operator==(const Edge &rhs) const {
            return m_source == rhs.m_source &&
                   m_target == rhs.m_target;
        }

        inline __host__ __device__ bool operator!=(const Edge &rhs) const {
            return !(*this == rhs);
        }

        inline __host__ __device__ bool operator>(const Edge &rhs) const {
            return m_source > rhs.m_source || (m_source == rhs.m_source && m_target > rhs.m_target);
        }

        inline __host__ __device__ bool operator<(const Edge &rhs) const {
            return m_source < rhs.m_source || (m_source == rhs.m_source && m_target < rhs.m_target);
        }

        /**
         * @param n number of vertices in graph.
         * @return true if edge is valid.
         */
        [[nodiscard]] inline __host__ __device__ bool is_valid(int_t n) const {
            return 0 <= m_source && m_source < m_target && m_target < n;
        }

        /**
         * @param vertex input vertex.
         * @return neighbor vertex of the input vertex.
         */
        [[nodiscard]] inline __host__ __device__ int_t opposite(int_t vertex) const {
            if (vertex == m_source) {
                return m_target;
            } else if (vertex == m_target) {
                return m_source;
            }
            while (true) {
                printf("Something wrong is going on\n");
            }}
    } Edge;

    /**
     * Printing method
     */
    std::ostream &operator<<(std::ostream &os, const Edge &edge);

    /**
     * Undirected graph presentation.
     */
    typedef struct Graph {
        using Ptr = std::shared_ptr<Graph>;

        Edge *m_edges{nullptr}; // M[m]
        int_t **m_incidence_matrix{nullptr}; // M[n, d...]
        int_t *m_degree{nullptr}; // M[n]
        int_t n{0};
        int_t m{0};

        Graph(const std::vector<Edge> &edges, int_t n);

        Graph(const Graph&) = delete;
        Graph(Graph&&) = delete;
        Graph& operator=(const Graph&) = delete;
        Graph&& operator=(Graph&&) = delete;

        ~Graph();

        bool operator==(const Graph &rhs) const;

        bool operator!=(const Graph &rhs) const;

        __device__ __host__ void vertices(const std::function<void(int_t)>& apply) const {
            for(int_t i = 0; i < n; i++) {
                apply(i);
            }
        }

        __device__ __host__ void edges(const std::function<void(int_t)>& apply) const {
            for(int_t i = 0; i < m; i++) {
                apply(i);
            }
        }
    } Graph;

    /**
     * Printing method
     */
    std::ostream &operator<<(std::ostream &os, const Graph &graph);
}