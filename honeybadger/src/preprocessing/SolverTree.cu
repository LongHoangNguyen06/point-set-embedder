#include <ranges>
#include <utility>

#include "honeybadger/preprocessing/SolverTree.cuh"
#include "honeybadger/JSON.cuh"

namespace honeybadger::solver_tree {
    SolverTree::SolverTree(parameters::AlgorithmParameters parameter,
                           Solution::Ptr root_solution,
                           const std::string &json_string) : m_root_parameter(std::move(parameter)),
                                                           m_root_solution(std::move(root_solution)),
                                                           m_json_string(json_string){
        if (m_root_parameter.m_mode == parameters::IMPROVE) {
            std::vector<Point> assignment = json::readAssignment(json_string);
            m_root_solution->assign(assignment);
        }
    }

    void SolverTree::solve() {
        construct_tree();
        if (m_preprocessing_tree.empty()) {
            solve(m_root_parameter, m_root_solution);
        } else {
            solve_leaves();
            propagate_back_to_root();
        }
    }

    void SolverTree::construct_tree() {
        LOGGING << "Start constructing solver tree" << std::endl;
        for (auto preprocessing : m_root_parameter.m_preprocessing_order) {
            switch (preprocessing) {
                case parameters::IGNORE_EDGES:
                    LOGGING << "Insert 'ignore edges' node." << std::endl;
                    this->createGenericLevel<preprocessing::ignore_edges::IgnoreEdges>();
                    break;
                case parameters::HINTS:
                    LOGGING << "Insert 'assignment hints' node." << std::endl;
                    this->createGenericLevel<preprocessing::assignment_hints::AssignmentHints>();
                    break;
                case parameters::IGNORE_VERTICES:
                    LOGGING << "Insert 'ignore vertices' node." << std::endl;
                    this->createGenericLevel<preprocessing::ignore_vertices::IgnoreVertices>();
                    break;
                case parameters::IGNORE_POINTS:
                    LOGGING << "Insert 'ignore points' node." << std::endl;
                    this->createGenericLevel<preprocessing::ignore_points::IgnorePoints>();
                    break;
                default:
                    std::exit(1);
            }
        }
    }

    void SolverTree::solve_leaves() {
        // Step 7: Solve the problem on sub solutions of the last preprocessing level
        LOGGING << "Start solving leaves solutions" << std::endl;
        for (const auto& preprocessing_leaf: m_preprocessing_tree.back()) {
            for (size_t i = 0; i < preprocessing_leaf->m_output_solutions.size(); i++) {
                auto leaf_solution = preprocessing_leaf->m_output_solutions[i];
                auto leaf_parameter = preprocessing_leaf->m_output_parameters[i];
                solve(leaf_parameter, leaf_solution);
            }
        }
    }

    void SolverTree::propagate_back_to_root() {
        // Step 4: Propagate the solutions along the pipeline backward
        LOGGING_NL << "All leaf solutions are finished. Start propagating back to root" << std::endl;
        for (auto & level : std::ranges::reverse_view(m_preprocessing_tree)) {
            for (const auto& node : level) {
                node->backward();
            }
        }
        LOGGING << "Propagation to root finished. Solving successful" << std::endl;
    }

    void SolverTree::solve(parameters::AlgorithmParameters &parameter, Solution::Ptr solution) {
        // Optimizing algorithm
        switch (parameter.m_algorithm) {
            case parameters::RANDOM:
                solveRandom(parameter, solution);
                break;
            case parameters::NEAREST_NEIGHBOR:
                solveInteractiveAll(solution);
                break;
            default:
                std::exit(1);
        }
    }

    void SolverTree::solveRandom(parameters::AlgorithmParameters &parameter, Solution::Ptr solution) {
        auto seed = msSinceEpoch();
        switch (parameter.m_mode) {
            case parameters::INITIALIZE:
                algorithms::random_search::initialize(solution, seed);
                break;
            case parameters::IMPROVE:
                algorithms::random_search::improve(solution, seed);
                break;
            case parameters::INITIALIZE_AND_IMPROVE:
                algorithms::random_search::initializeAndImprove(solution, seed);
                break;
            default:
                std::exit(1);
        }
    }

    void SolverTree::solveInteractiveAll(Solution::Ptr solution) {
        algorithms::nearest_neighbors::initialize(std::move(solution));
    }
}