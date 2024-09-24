#include "honeybadger/datastructure/Problem.cuh"
#include <algorithm>
#include <utility>

namespace honeybadger {
    Problem::Problem(Graph::Ptr graph, const std::vector<Point> &points, int_t id) {
        if (hasDuplicate(points)) {
            throw_error("There are duplicates in points set " << std::endl, std::invalid_argument)
        }
        assert(!hasDuplicate(points));

        this->p = static_cast<int_t>(points.size());
        this->m_graph = std::move(graph);
        this->m_id = id;

        if (this->m_graph->n > this->p) {
            throw_error("Too few points. Graph has " << this->m_graph->n << " nodes but there are only "
                                                     << this->p
                                                     << " points.", std::invalid_argument)
        }
        for (const auto &point: points) {
            if (!point.is_valid()) {
                throw_error("Point is not valid " << point, std::invalid_argument)
            }
        }

        gpu_error_check(robust_cuda_malloc_managed(this->m_points, this->p))
        for (int_t i = 0; i < points.size(); i++) {
            this->m_points[i] = points[i];
        }
    }

    Problem::~Problem() {
        gpu_error_check(cudaFree(this->m_points))
    }

    std::ostream &Problem::operator<<(std::ostream &os) const {
        os << "Problem{";
        os << std::endl << "\t" << "id=" << this->m_id;
        os << std::endl << "m_points [" << this->p << "]";
        for (int_t i = 0; i < this->p; i++) {
            os << std::endl << "\t" << i << ": " << this->m_points[i];
        }
        os << std::endl << "}";
        return os;
    }

    bool Problem::operator==(const Problem &rhs) const {
        if (p != rhs.p) {
            LOGGING << "P not equal" << std::endl;
            return false;
        }
        if (*m_graph != *rhs.m_graph) {
            LOGGING << "Graph not equal" << std::endl;
            return false;
        }
        for (int_t i = 0; i < p; i++) {
            if (getPointAt(i) != rhs.getPointAt(i)) {
                LOGGING << "Point not equal" << std::endl;
                return false;
            }
        }
        return true;
    }

    bool Problem::operator!=(const Problem &rhs) const {
        return !(rhs == *this);
    }
}