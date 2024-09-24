#pragma once
#include <map>
#include <string>
#include <functional>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/datastructure/Problem.cuh"
#include "honeybadger/datastructure/Graph.cuh"
#include "Preprocessing.cuh"
#include "honeybadger/algorithms/Algorithms.cuh"
#include "honeybadger/Parameters.cuh"

namespace honeybadger::solver_tree {
    /**
     * Solver class which:
     * - splits the input solution into sub-problems
     * - pre-processes the sub-problems
     * - solves the sub-problems and combine them back to original input solution.
     */
    class SolverTree {
    private:
        parameters::AlgorithmParameters m_root_parameter;
        Solution::Ptr m_root_solution;
        std::vector<std::vector<preprocessing::BasePreprocessing::Ptr>> m_preprocessing_tree{};
        std::string m_json_string;
    private:
        /**
         * Expand tree with new pre-processing.
         * @param preprocessorInitiator take a parameter, a solution as input and produce preprocess.
         */
        template <class T>
        void createGenericLevel() {
            std::vector<preprocessing::BasePreprocessing::Ptr> new_preprocessing_leaves;
            if (m_preprocessing_tree.empty()) { // Tree empty
                new_preprocessing_leaves.push_back(std::make_shared<T>(m_root_solution, m_root_parameter));
            } else { // Tree not empty, iterate leaves of current tree
                for (const auto &preprocessing_leaf: m_preprocessing_tree.back()) {
                    for (size_t i = 0; i < preprocessing_leaf->m_output_solutions.size(); i++) {
                        auto current_solution_leaf = preprocessing_leaf->m_output_solutions[i];
                        auto current_parameter_leaf = preprocessing_leaf->m_output_parameters[i];

                        // For each leaf, create new child
                        new_preprocessing_leaves.push_back(std::make_shared<T>(current_solution_leaf,
                                                                               current_parameter_leaf));
                    }
                }
            }
            m_preprocessing_tree.push_back(new_preprocessing_leaves); // Create new level in tree
            for(auto & leaf : new_preprocessing_leaves) { // Run new leaves
                leaf->forward();
            }
        }

        void construct_tree();

        void solve_leaves();

        void propagate_back_to_root();

        /**
         * Solve a leaf-problem.
         */
        static void solve(parameters::AlgorithmParameters &parameter, Solution::Ptr solution);

        /**
         * Solve a leaf-problem with ILP.
         */
        static void solveWithIlp(parameters::AlgorithmParameters &parameter, Solution::Ptr solution);

        /**
         * Solve a leaf-problem with random algorithm.
         */
        static void solveRandom(parameters::AlgorithmParameters &parameter, Solution::Ptr solution);

        /**
         * Solve a leaf-problem with interactive algorithm.
         */
        static void solveInteractiveAll(Solution::Ptr solution);
    public:
        SolverTree(parameters::AlgorithmParameters parameter, Solution::Ptr root_solution, const std::string& json_string);

        void solve();
    };
}