#include "honeybadger/algorithms/NearestNeighbors.cuh"
#include <deque>
#include <algorithm>

namespace honeybadger::algorithms::nearest_neighbors {
    class Algorithm {
        honeybadger::Solution::Ptr m_solution;
        std::vector<int_t> m_num_assigned_neighbors{};

        /**
         * Return the nearest point to all other points of neighbors.
         */
        int_t nearestPointOf(const std::vector<Point> &points) {
            std::vector<double> distances(m_solution->data.p, 0);

            // Iterate every candidate point and compute distance of the candidate point
            for (int_t i = 0; i < m_solution->data.p; i++) {

                // Check if candidate is assigned or is crossed
                if (m_solution->data.isPointAssigned(i) || m_solution->data.isPointCrossed(i)) {
                    distances[i] = std::numeric_limits<double>::infinity();
                    continue;
                }

                // Check if inserting a vertex into candidate would cross any assigned vertex
                bool found = false;
                for (int_t j = 0; j < *m_solution->data.m_num_available_points; j++) {
                    if(m_solution->data.m_available_points_indices[j] == i) {
                        found = true;
                        if (m_solution->data.m_available_points_cross_assigned_points[j]) {
                            distances[i] = std::numeric_limits<double>::infinity();
                            break;
                        }
                    }
                }
                assert(found);

                // Compute the sum distance of candidate to points of reference
                if (distances[i] != std::numeric_limits<double>::infinity()) {
                    // Is really a possible point, compute distance
                    double x2 = m_solution->data.m_points[i].m_x;
                    double y2 = m_solution->data.m_points[i].m_y;
                    for (Point point: points) {
                        double x1 = point.m_x;
                        double y1 = point.m_y;
                        distances[i] += std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
                    }
                }
            }
            auto min_point= std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
            if (distances[min_point] == std::numeric_limits<double>::infinity()) {
                throw_error("Nearest neighbors algorithm can not find any further point. Please try again", NotSatisfiableException)
            }
            return min_point;
        }

        /**
         * Mark the inserted vertex as inserted so the algorithm won't visit it again.
         * Also, increase the number of assigned neighbors for each neighbor of the inserted vertex.
         */
        void updateStatusOfAssignedVertex(int_t v) {
            m_num_assigned_neighbors[v] = std::numeric_limits<int_t>::min(); // Make sure v will not be proposed, again.
            for (auto d = 0; d < m_solution->data.m_degree[v]; d++) {
                auto adj_idx = m_solution->data.m_incidence_matrix[v][d];
                auto adj = m_solution->data.m_edges[adj_idx];
                auto u = adj.opposite(v);
                if (m_solution->data.isVertexVisible(v)) {
                    m_num_assigned_neighbors[u]++;
                }
            }

        }

        /**
         * Insert the vertex and increase neighbor counts of each vertex' neighbors.
         */
        void insertVertex(int_t v, int_t point_idx) {
            m_solution->insert(v, point_idx);
            updateStatusOfAssignedVertex(v);
        }

    public:
        /**
         * Constructor
         */
        explicit Algorithm(honeybadger::Solution::Ptr solution) : m_solution(solution), m_num_assigned_neighbors(m_solution->data.n, 0) {
            for (int_t v = 0; v < m_solution->data.n; v++) {
                if(m_solution->data.isVertexVisible(v)) {
                    this->updateStatusOfAssignedVertex(v);
                }
            }
            LOGGING << "Start nearest neighbor. Input problem: " << SUMMARY(m_solution->data) << std::endl;
        }

        /**
         * Start the algorithm
         */
        void start() {
            for(int_t i = 0; i < m_solution->data.n; i++) {
                if (m_solution->data.isVertexVisible(i)) {
                    LOGGING << "Vertex " << i << " is already assigned to point " << m_solution->data.m_assignments[i]
                            << ". Skip!" << std::endl;
                    continue;
                }
            }

            for(int_t i = 0; i < m_solution->data.n; i++) {
                // Get vertex with most assigned neighbors
                auto v = std::distance(m_num_assigned_neighbors.begin(),
                                       std::max_element(m_num_assigned_neighbors.begin(),
                                                        m_num_assigned_neighbors.end()));
                if (m_solution->data.isVertexVisible(v)) {
                    continue;
                }

                // Get neighbors' points
                std::vector<Point> neighbor_points;
                for (auto d = 0; d < m_solution->data.m_degree[v]; d++) {
                    auto adj_idx = m_solution->data.m_incidence_matrix[v][d];
                    auto adj = m_solution->data.m_edges[adj_idx];
                    auto u = adj.opposite(v);
                    if (m_solution->data.isVertexVisible(u)) {
                        neighbor_points.push_back(m_solution->data.m_points[m_solution->data.m_assignments[u]]);
                    }
                }

                // Get nearest points to all neighbors
                m_solution->findBestPoint(v);
                int_t nearest_point = nearestPointOf(neighbor_points);
                LOGGING << "Assigning vertex " << v << " to point " << nearest_point << std::endl;
                this->insertVertex(v, nearest_point);
            }
        }
    };

    void initialize(honeybadger::Solution::Ptr solution) {
        LOGGING << "Nearest neighbor solver will try to find a feasible solution." << std::endl;
        Algorithm algorithm(solution);
        algorithm.start();
    }
}
