#include <utility>

#include "honeybadger/preprocessing/IgnoreVertices.cuh"

namespace honeybadger::preprocessing::ignore_vertices {
    IgnoreVertices::IgnoreVertices(Solution::Ptr solution, const parameters::AlgorithmParameters &parameter)
            : BasePreprocessing(
            std::move(solution), parameter), m_ignored_input_vertices(parameter.m_ignored_vertices) {
    }

    void IgnoreVertices::forward() {
        // See which vertices will be kept
        LOGGING << "Start 'ignore vertices'. Input problem: " << SUMMARY(m_input_solution->data) << std::endl;

        // Compute mapping old vertices and new vertices
        for (int_t input_idx = 0, output_idx = 0; input_idx < m_input_solution->data.n; input_idx++) {
            if (!m_ignored_input_vertices.contains(input_idx)) {
                m_input_vertices_to_output_vertices[input_idx] = output_idx;
                m_output_vertices_to_input_vertices[output_idx] = input_idx;
                output_idx++;
            } else {
                LOGGING << "Ignore vertex " << input_idx << std::endl;
            }
        }

        // Compute edge
        for (int_t input_edge_idx = 0; input_edge_idx < m_input_solution->data.m; input_edge_idx++) {
            auto input_edge = m_input_solution->data.m_edges[input_edge_idx];
            if (!m_ignored_input_vertices.contains(input_edge.m_source) &&
                !m_ignored_input_vertices.contains(input_edge.m_target)) {
                m_output_edges.emplace_back(m_input_vertices_to_output_vertices[input_edge.m_source],
                                            m_input_vertices_to_output_vertices[input_edge.m_target]);
            } else {
                LOGGING << "Ignoring edge " << input_edge << std::endl;
            }
        }

        BasePreprocessing::forward();

        LOGGING << "Finish 'ignore vertices'. Output problem: " << SUMMARY(m_output_solution->data) << std::endl;
    }

    void IgnoreVertices::backward() {
        BasePreprocessing::backward();
        LOGGING << "Propagate from 'ignore vertices' to parent" << std::endl;
    }

    void IgnoreVertices::transferAssignmentFromInputToOutput() {
        // Transfer assignment from input solution to output solution
        for (int_t input_idx = 0; input_idx < m_input_solution->data.n; input_idx++) {
            if (m_input_solution->data.isVertexVisible(input_idx)) {
                m_output_solution->insert(m_input_vertices_to_output_vertices[input_idx],
                                          m_input_solution->data.m_assignments[input_idx]);
            }
        }
    }

    void IgnoreVertices::transferAssignmentFromOutputToInput() {
        // Reset old solution
        for (int_t i = 0; i < m_input_solution->data.n; i++) {
            if (m_input_solution->data.isVertexVisible(i)) {
                m_input_solution->remove(i);
            }
        }

        // Transfer assignment from output solution back to input solution
        for (int_t output_idx = 0; output_idx < m_output_solutions.front()->data.n; output_idx++) {
            if (m_output_solution->data.isVertexVisible(output_idx)) {
                m_input_solution->insert(m_output_vertices_to_input_vertices[output_idx],
                                         m_output_solution->data.m_assignments[output_idx]);
            }
        }

        // Find points for ignored vertices
        for (int_t i = 0; i < m_input_solution->data.n; i++) {
            if (m_ignored_input_vertices.contains(i)) {
                m_input_solution->findBestPoint(i);
                if (*m_input_solution->data.m_num_best_points == 0) {
                    throw_error("Can not find any point left for the ignored vertex " << i, NotSatisfiableException)
                }
                m_input_solution->insert(i, m_input_solution->data.m_best_points_indices[0]);
            }
        }
    }

    void IgnoreVertices::produceOutputSolutions() {
        m_output_graph = std::make_shared<Graph>(m_output_edges, m_output_vertices_to_input_vertices.size());

        std::vector<Point> output_points(m_input_solution->data.m_points, m_input_solution->data.m_points + m_input_solution->data.p);
        m_output_problem = std::make_shared<Problem>(m_output_graph, output_points);

        m_output_solution = std::make_shared<Solution>(m_output_problem);
        this->m_output_solutions.push_back(m_output_solution);
    }

    void IgnoreVertices::produceOutputParameters() {
        m_output_parameter = parameters::AlgorithmParameters(this->m_input_parameter);

        // Empty ignored_input_vertices
        m_output_parameter.m_ignored_vertices.clear();

        // Augment ignored_edges
        m_output_parameter.m_ignored_edges.clear();
        for (auto &ignored_input_edge: m_input_parameter.m_ignored_edges) {
            // If the ignored edge contains an ignored vertex, skip.
            if (m_input_parameter.m_ignored_vertices.contains(ignored_input_edge.first) ||
                m_input_parameter.m_ignored_vertices.contains(ignored_input_edge.second)) {
                continue;
            }
            // Else, map input edge to output edge
            m_output_parameter.m_ignored_edges.insert(
                    {m_input_vertices_to_output_vertices[ignored_input_edge.first],
                     m_input_vertices_to_output_vertices[ignored_input_edge.second]});
        }

        // Augment hints
        m_output_parameter.m_hints.clear();
        for (auto &input_hint: m_input_parameter.m_hints) {
            // If the vertex of the hint is ignored, skip
            if (m_input_parameter.m_ignored_vertices.contains(input_hint.first)) {
                continue;
            }
            // Else map to a new hint
            m_output_parameter.m_hints.insert(
                    {m_input_vertices_to_output_vertices[input_hint.first],
                     input_hint.second});
        }

        // Place parameters in buffer
        m_output_parameters.push_back(m_output_parameter);
    }
}