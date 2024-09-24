#pragma once

#include "honeybadger/Utils.cuh"
#include "Graph.cuh"
#include "honeybadger/Geometry.cuh"
#include <vector>
#include <ostream>
#include <memory>

namespace honeybadger {
    /**
     * Problem class.
     */
    typedef struct Problem {
        using Ptr = std::shared_ptr<Problem>;

        Graph::Ptr m_graph{}; // Graph of the problem
        int_t p{}; // Number of points
        Point *m_points{}; // M[p]
        int_t m_id{};

        /**
         * Constructor of problem
         * @param graph input graph
         * @param points input points
         * @param id contest's id
         */
        Problem(Graph::Ptr graph, const std::vector<Point> &points, int_t id = 0);

        Problem(const Problem&) = delete;
        Problem(Problem&&) = delete;
        Problem& operator=(const Problem&) = delete;
        Problem&& operator=(Problem&&) = delete;

        ~Problem();

        [[nodiscard]] inline __device__ __host__ Point getPointAt(int_t idx) const {
            return this->m_points[idx];
        }

        bool operator==(const Problem &rhs) const;

        bool operator!=(const Problem &rhs) const;

        std::ostream &operator<<(std::ostream &os) const;
    } Problem;
}