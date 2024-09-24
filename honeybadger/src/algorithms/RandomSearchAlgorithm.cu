#include <algorithm>
#include <execution>
#include <random>
#include <csignal>
#include <utility>

#include "honeybadger/algorithms/RandomSearchAlgorithm.cuh"
#include "honeybadger/Utils.cuh"

namespace honeybadger::algorithms::random_search {
    class Algorithm {
        Solution::Ptr m_solution;
        int_t m_rounds{1};      // Documenting how many rounds were run.
        int_t m_iterations{1};  // Documenting how many iterations were ran.
        std::vector<int_t>
                m_nodes{};  // Ordering in which the vertices will be inserted.
        int_t *m_best_assignment{};  // M[n] vector stores the best assigment found.
        int_t m_best_objective_value{
                std::numeric_limits<int_t>::max()};  // Marking the best found objective
        // value.
        int_t m_iterations_without_improvement{
                0};  // Marking how many iterations have been ran without improving the
        // objective value.
        int_t m_iterations_with_improvement{
                0};  // Marking how many consecutive iterations have been ran that improve
        // the objective value.

        // Time measuring algorithm
        int_t m_time_start{secondsSinceEpoch()};  // Start time mark of the algorithm
        int_t m_time_elapsed_seconds{1};  // Mark how many seconds have been elapsed
        // since the beginning of the algorithm.
        int_t m_seed{0};        // Random seed
        std::mt19937 m_random_engine{0};
        int_t m_neighborhood_size{2}; // How many nodes will be reinserted at once.
        int_t m_rollback{0}; // Indicates how many times the algorithm must rollback to a previous valid solution. The higher this number, the more probably that the algorithm is running against a wall.
        std::optional<int_t> m_user_defined_neighborhood{std::nullopt};
        /**
         * Update the time used to search.
         */
        void updateTime() {
            m_time_elapsed_seconds = secondsSinceEpoch() - m_time_start;
        }

    public:
        volatile bool m_interrupted = false; // Flag indicates if user wants to stop the program

        /**
         * Constructor
         * @param solution input solution
         * @param seed input seed
         */
        explicit Algorithm(Solution::Ptr solution, int_t seed, std::optional<int_t> neighborhood)
                : m_solution(solution), m_seed(seed), m_random_engine(std::mt19937(m_seed)), m_user_defined_neighborhood(neighborhood) {
            gpu_error_check(robust_cuda_malloc_managed(
                    m_best_assignment, solution->data.n))
            for (int_t i = 0; i < m_solution->data.n;
                 i++) {
                m_best_assignment[i] = NON_VALID_ASSIGNMENT;
            }
            for (int_t i = 0; i < m_solution->data.n;
                 i++) {
                m_nodes.push_back(i);
            }
            m_time_start = secondsSinceEpoch();
            LOGGING << "Start random search. Input problem: " << SUMMARY(m_solution->data) << std::endl;
        }

        /**
         * Destructor
         */
        ~Algorithm() { cudaFree(m_best_assignment); }

        /**
         * Initialize a valid drawing.
         */
        void phase1() {
            LOGGING << "Starting phase 1. Inserting vertex for vertex, sorting by degree" << std::endl;
            // Sort nodes by degree
            std::sort(std::execution::par_unseq, m_nodes.begin(), m_nodes.end(), [&](int_t a, int_t b) {
                return m_solution->data.m_degree[a] > m_solution->data.m_degree[b];
            });

            // Insert one node after each other
            for (int_t i: m_nodes) {
                if (!m_solution->data.isVertexVisible(i)) {
                    insertNode(i);
                    printProgress(1);
                    m_best_objective_value = *m_solution->data.m_objective_value;
                }
            }

            m_rounds++;
        }

        /**
         * Reinserting node for node, ordered by nodes with most crossings.
         */
        void phase2() {
            LOGGING_NL << "Starting phase 2, inserting vertex for vertex, sorting by crossings. Press Ctrl + C to stop" << std::endl;
            while (!m_interrupted) {
                // Sort nodes by crossings
                std::sort(std::execution::par_unseq, m_nodes.begin(), m_nodes.end(),
                          [&](int_t a, int_t b) {
                              return m_solution->data.m_node_crossings[a] >
                                     m_solution->data.m_node_crossings[b];
                          });

                // Reinsert nodes
                for (int_t i: m_nodes) {
                    auto before = *m_solution->data.m_objective_value;
                    this->reinsertNode(i);
                    if (*m_solution->data.m_objective_value > before) {
                        throw_error("Reinserting worsened from " << before << " to " << *m_solution->data.m_objective_value,
                                    std::runtime_error)
                    }
                    printProgress(2);
                    if (m_interrupted) {
                        break;
                    }
                }

                m_best_objective_value = *m_solution->data.m_objective_value;
            }
            m_interrupted = false;
        }

        /**
         * Reinsert k nodes at once.
         */
        void phase3() {
            LOGGING_NL << "Starting phase 3, local search with dynamic neighborhood's size. Press Ctrl + C to stop" << std::endl;
            this->saveBestSolution();
            while (!m_interrupted && m_rounds++) {
                for (int_t i = 0; i < m_nodes.size(); i += m_neighborhood_size) {
                    // Remove vertices we want to reinsert
                    int_t n_nodes_removed = std::min(m_neighborhood_size, m_solution->data.n - i - 1);
                    std::vector<int_t> removed_vertices(n_nodes_removed);
                    std::vector<int_t> removed_vertices_points(n_nodes_removed);
                    for (int_t j = i, running_var = 0l; running_var < n_nodes_removed; j++, running_var++) {
                        auto removed_vertex = m_nodes[j];
                        removed_vertices[running_var] = removed_vertex;
                        removed_vertices_points[running_var] = m_solution->data.m_assignments[removed_vertex];
                        m_solution->remove(removed_vertex);
                    }

                    // Reinsert nodes
                    try {
                        for (int_t j = i, running_var = 0l; running_var < n_nodes_removed; j++, running_var++) {
                            this->insertNode(m_nodes[j]);
                        }

                        // Store if found new best solution.
                        if (*m_solution->data.m_objective_value < m_best_objective_value) {
                            this->saveBestSolution();
                        }

                        // Adjust search space's neighborhood
                        this->printProgress(3);
                        this->adjustNeighborhoodSize();
                    } catch (
                            NotSatisfiableException &e) { // Roll back to previous solution if new solution is not valid
                        for (int_t removed_vertex: removed_vertices) {
                            if (m_solution->data.isVertexVisible(removed_vertex)) {
                                m_solution->remove(removed_vertex);
                            }
                        }
                        for (int_t idx = 0; idx < n_nodes_removed; idx++) {
                            m_solution->insert(removed_vertices[idx],
                                               removed_vertices_points[idx]);
                        }
                        m_rollback++;
                    }
                    if (m_interrupted) {
                        break;
                    }
                }

                // Reshuffle the vertices
                std::shuffle(m_nodes.begin(), m_nodes.end(), m_random_engine);
            }
            this->restoreBestSolution();
        }

    private:
        /**
         * Reinsert a node into a a new better point.
         * @param node node to be reinserted.
         * @param removeNodeFirst flag indicates if the input node should be removed before reinserted.
         */
        void insertNode(int_t node) {
            // Find new best point for the vertex
            auto best_points = m_solution->findBestPoint(node);
            if (best_points.empty()) {
                throw_error("Can not find any point to reinsert " << node, NotSatisfiableException)
            }

            // Choose any random best point
            std::uniform_int_distribution<int_t> random_dist(0, best_points.size() - 1);
            auto random_best_point_idx = random_dist(m_random_engine);
            // Reinsert vertex into the best new point
            m_solution->insert(node, best_points[random_best_point_idx]);
        }

        /**
         * Reinsert a node into a a new better point.
         * @param node node to be reinserted.
         * @param removeNodeFirst flag indicates if the input node should be removed before reinserted.
         */
        void reinsertNode(int_t node) {
            m_solution->remove(node);

            insertNode(node);
        }

        /**
         * Store the current best solution into buffer.
         */
        void saveBestSolution() {
            m_best_objective_value = *m_solution->data.m_objective_value;
            gpu_error_check(robust_cuda_memcpy(m_solution->data.m_assignments, m_best_assignment, m_solution->data.n))
        }

        /**
         * Restore the best solution from buffer.
         */
        void restoreBestSolution() {
            for(int_t i = 0; i < m_solution->data.n; i++) {
                m_solution->remove(i);
            }
            for(int_t i = 0; i < m_solution->data.n; i++) {
                m_solution->insert(i, m_best_assignment[i]);
            }
        }

        /**
         * Print the progress.
         */
        void printProgress(int_t phase) {
            LOGGING << "\x1b[2K\r";
            LOGGING << "Phase=" << phase
              << " round=" << m_rounds
              << " rollbacks=" << m_rollback
              << " time=" << m_time_elapsed_seconds
              << " iterations=" << m_iterations++
              << " neighborhood=" << m_neighborhood_size
              << " best=" << m_best_objective_value
              << " crossings=" << *m_solution->data.m_objective_value;
            updateTime();
        }

        /**
         * Adjust the number of nodes to be reinserted at once in phase 3.
         */
        void adjustNeighborhoodSize() {
            if (*m_solution->data.m_objective_value >= m_best_objective_value) {
                m_iterations_without_improvement++;
                m_iterations_with_improvement = 0;
            } else {
                m_iterations_with_improvement++;
                m_iterations_without_improvement = 0;
            }
            if (m_iterations_with_improvement == 3) {
                m_iterations_with_improvement = 0;
                m_neighborhood_size = std::max(1l, m_neighborhood_size - decreaseNeighborhoodSize());
            }
            if (m_iterations_without_improvement == 10) {
                m_iterations_without_improvement = 0;
                m_neighborhood_size = std::min(maxNeighborhoodSize(),
                                               m_neighborhood_size + increaseNeighborhoodSize());
            }
        }

        [[nodiscard]] int_t maxNeighborhoodSize() const {
            if (m_user_defined_neighborhood.has_value()) {
                return std::numeric_limits<int_t>::max();
            }
            if (m_solution->data.n <= 10) {
                return 2;
            } else if (m_solution->data.n <= 15) {
                return 4;
            } else if (m_solution->data.n <= 30) {
                return 6;
            } else if (m_solution->data.n <= 50) {
                return 10;
            } else if (m_solution->data.n <= 100) {
                return 5;
            } else if (m_solution->data.n <= 200) {
                return 25;
            } else if (m_solution->data.n <= 500) {
                return 50;
            } else if (m_solution->data.n <= 1000) {
                return 100;
            } else if (m_solution->data.n <= 1500) {
                return 150;
            }
            return 200;
        }

        [[nodiscard]] int_t increaseNeighborhoodSize() const {
            if (m_solution->data.n <= 15) {
                return 1;
            } else if (m_solution->data.n <= 50) {
                return 2;
            } else if (m_solution->data.n <= 100) {
                return 3;
            } else if (m_solution->data.n <= 150) {
                return 4;
            } else if (m_solution->data.n <= 200) {
                return 5;
            } else if (m_solution->data.n <= 250) {
                return 6;
            } else if (m_solution->data.n <= 300) {
                return 7;
            } else if (m_solution->data.n <= 350) {
                return 8;
            } else if (m_solution->data.n <= 400) {
                return 9;
            } else if (m_solution->data.n <= 450) {
                return 10;
            } else if (m_solution->data.n <= 500) {
                return 11;
            } else if (m_solution->data.n <= 1000) {
                return 12;
            } else if (m_solution->data.n <= 1500) {
                return 13;
            }
            return 20;
        }

        [[nodiscard]] int_t decreaseNeighborhoodSize() const {
            if (m_solution->data.n <= 50) {
                return 1;
            } else if (m_solution->data.n <= 150) {
                return 2;
            } else if (m_solution->data.n <= 250) {
                return 3;
            } else if (m_solution->data.n <= 350) {
                return 4;
            } else if (m_solution->data.n <= 450) {
                return 5;
            } else if (m_solution->data.n <= 500) {
                return 6;
            } else if (m_solution->data.n <= 1000) {
                return 7;
            } else if (m_solution->data.n <= 1500) {
                return 8;
            }
            return 20;
        }
    };

    std::function<void(int)> shutdown_handler;

    void signalHandler(int signal) { shutdown_handler(signal); }

    void setupSignalHandler(Algorithm &algorithm) {
        // Set up shutdown handler.
        shutdown_handler = [&](int signal) {
            algorithm.m_interrupted = true;
        };
        signal(SIGINT, signalHandler);
    }

    void initialize(Solution::Ptr solution, int_t seed) {
        LOGGING << "Random search will only try to find a feasible solution" << std::endl;
        Algorithm algorithm(std::move(solution), seed, std::nullopt);
        setupSignalHandler(algorithm);
        algorithm.phase1();
    }

    void improve(Solution::Ptr solution, int_t seed, std::optional<int_t> neighborhood) {
        LOGGING << "Random search will only try to improve a feasible solution" << std::endl;
        Algorithm algorithm(std::move(solution), seed, neighborhood);
        setupSignalHandler(algorithm);
        algorithm.phase2();
        algorithm.phase3();
    }


    void initializeAndImprove(Solution::Ptr solution, int_t seed, std::optional<int_t> neighborhood) {
        LOGGING << "Random search will try to minimize crossings from scratch" << std::endl;
        Algorithm algorithm(std::move(solution), seed, neighborhood);
        setupSignalHandler(algorithm);
        algorithm.phase1();
        algorithm.phase2();
        algorithm.phase3();
    }
}  // namespace honeybadger::algorithms::random_search
