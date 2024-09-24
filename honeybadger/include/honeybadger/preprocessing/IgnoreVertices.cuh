#pragma once

#include <map>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "BasePreprocessing.cuh"

namespace honeybadger::preprocessing::ignore_vertices {
    /**
     * Delete vertices in the input graph.
     */
    class IgnoreVertices : public BasePreprocessing {
    private:
        std::map<int_t, int_t> m_input_vertices_to_output_vertices;
        std::map<int_t, int_t> m_output_vertices_to_input_vertices;
        std::set<int_t> m_ignored_input_vertices;
        std::vector<Edge> m_output_edges;

        Graph::Ptr m_output_graph{};
        Problem::Ptr m_output_problem{};
        Solution::Ptr m_output_solution{};
        parameters::AlgorithmParameters m_output_parameter;
    protected:
        void transferAssignmentFromInputToOutput() override;

        void transferAssignmentFromOutputToInput() override;

        void produceOutputSolutions() override;

        void produceOutputParameters() override;
    public:
        IgnoreVertices(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter);

        void forward() override;

        void backward() override;
    };
}