#include <utility>

#include "honeybadger/preprocessing/IgnoreEdges.cuh"

namespace honeybadger::preprocessing::ignore_edges {
    IgnoreEdges::IgnoreEdges(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter) : BasePreprocessing(
            std::move(solution), parameter) {
        for (auto edge: parameter.m_ignored_edges) {
            m_ignored_edges.insert(Edge(edge.first, edge.second));
        }}

    void IgnoreEdges::forward() {
        LOGGING << "Start 'ignore edges'. Input problem: " << SUMMARY(m_input_solution->data) << std::endl;

        // See which edges will be kept
        for(int_t idx = 0; idx < m_input_solution->data.m; idx++) {
            if (!m_ignored_edges.contains(m_input_solution->data.m_edges[idx])) {
                m_output_edges.push_back(m_input_solution->data.m_edges[idx]);
            } else {
                LOGGING << "Ignore edge " <<  m_input_solution->data.m_edges[idx] << std::endl;
            }
        }
        BasePreprocessing::forward();

        LOGGING << "Finish 'ignore edges'. Output problem: " << SUMMARY(m_output_solution->data) << std::endl;
    }

    void IgnoreEdges::backward() {
        BasePreprocessing::backward();
        LOGGING << "Propagate from 'ignore edges' to parent" << std::endl;
    }

    void IgnoreEdges::transferAssignmentFromInputToOutput() {
        for(int_t i = 0; i < m_input_solution->data.n; i++) {
            if(m_input_solution->data.isVertexVisible(i)) {
                m_output_solution->insert(i, m_input_solution->data.m_assignments[i]);
            }
        }
    }

    void IgnoreEdges::transferAssignmentFromOutputToInput() {
        // Reset old solution
        for(int_t i = 0; i < m_input_solution->data.n; i++) {
            if(m_input_solution->data.isVertexVisible(i)) {
                m_input_solution->remove(i);
            }
        }
        // Transfer assignment from output solution back to input solution
        for(int_t i = 0; i < m_input_solution->data.n; i++) {
            if(m_output_solution->data.isVertexVisible(i)) {
                m_input_solution->insert(i, m_output_solution->data.m_assignments[i]);
            }
        }
    }

    void IgnoreEdges::produceOutputSolutions() {
        m_output_graph = std::make_shared<Graph>(m_output_edges, m_input_solution->data.n);
        std::vector<Point> output_points(m_input_solution->data.m_points, m_input_solution->data.m_points + m_input_solution->data.p);
        m_output_problem = std::make_shared<Problem>(m_output_graph, output_points);
        m_output_solution = std::make_shared<Solution>(m_output_problem);
        this->m_output_solutions.push_back(m_output_solution);

    }

    void IgnoreEdges::produceOutputParameters() {
        this->m_output_parameters.emplace_back(this->m_input_parameter);
    }
}