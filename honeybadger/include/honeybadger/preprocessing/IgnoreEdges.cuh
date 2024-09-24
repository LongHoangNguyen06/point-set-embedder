#pragma once

#include <map>
#include <set>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "BasePreprocessing.cuh"

namespace honeybadger::preprocessing::ignore_edges {
    /**
     * Delete edges in the input graph.
     */
    class IgnoreEdges : public BasePreprocessing {
    private:
        std::set<Edge> m_ignored_edges;
        std::vector<Edge> m_output_edges;
        Graph::Ptr m_output_graph{};
        Problem::Ptr m_output_problem{};
        Solution::Ptr m_output_solution{};
    protected:
        void transferAssignmentFromInputToOutput() override;

        void transferAssignmentFromOutputToInput() override;

        void produceOutputSolutions() override;

        void produceOutputParameters() override;
    public:
        IgnoreEdges(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter);

        void forward() override;

        void backward() override;
    };
}