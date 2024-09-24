#include "honeybadger/preprocessing/IgnorePoints.cuh"
#include "honeybadger/honeybadger.cuh"

namespace honeybadger::preprocessing::ignore_points {
    void IgnorePoints::transferAssignmentFromInputToOutput() {
        for (int_t input_vertex = 0; input_vertex < m_input_solution->data.n; input_vertex++) {
            if (m_input_solution->data.isVertexVisible(input_vertex)) {
                auto input_point = m_input_solution->data.m_assignments[input_vertex];
                if (m_input_parameter.m_ignored_points.contains(input_point)) {
                    throw_error("Try to ignore a point that was assigned, illegal!", std::invalid_argument)
                }
                auto output_point = m_input_points_to_output_points[input_point];
                m_input_solution->insert(input_vertex, output_point);
            }
        }
    }

    void IgnorePoints::transferAssignmentFromOutputToInput() {
        for (int_t input_vertex = 0; input_vertex < m_input_solution->data.n; input_vertex++) {
            if (m_input_solution->data.isVertexVisible(input_vertex)) {
                m_input_solution->remove(input_vertex);
            }
        }
        for (int_t input_vertex = 0; input_vertex < m_input_solution->data.n; input_vertex++) {
            auto output_point = m_output_solution->data.m_assignments[input_vertex];
            auto input_point = m_output_points_to_input_points[output_point];
            m_input_solution->insert(input_vertex, input_point);
        }
    }

    void IgnorePoints::produceOutputSolutions() {
        this->m_output_problem = std::make_shared<Problem>(m_input_solution->m_graph, m_kept_points,
                                              m_input_solution->m_problem->m_id);
        this->m_output_solution = std::make_shared<Solution>(m_output_problem);
        this->m_output_solutions.push_back(m_output_solution);
    }

    void IgnorePoints::produceOutputParameters() {
        m_output_parameter = parameters::AlgorithmParameters(this->m_input_parameter);
        m_output_parameter.m_ignored_points.clear();
        m_output_parameter.m_hints.clear();
        for (auto &hint: m_input_parameter.m_hints) {
            if (m_input_parameter.m_ignored_points.contains(hint.second)) {
                throw_error("Hints contains ignored point, illegal!", std::invalid_argument)
            }
            m_output_parameter.m_hints[hint.first] = m_input_points_to_output_points[hint.second];
        }
        this->m_output_parameters.push_back(m_output_parameter);
    }

    IgnorePoints::IgnorePoints(Solution::Ptr solution, const parameters::AlgorithmParameters &parameter)
            : BasePreprocessing(solution, parameter) {

    }

    void IgnorePoints::forward() {
        LOGGING << "Start 'ignore points'. Input problem: " << SUMMARY(m_input_solution->data) << std::endl;
        for (int_t input_point = 0, output_point = 0; input_point < m_input_solution->data.p; input_point++) {
            if (!this->m_input_parameter.m_ignored_points.contains(input_point)) {
                m_input_points_to_output_points[input_point] = output_point;
                m_output_points_to_input_points[output_point] = input_point;
                output_point++;
                this->m_kept_points.push_back(m_input_solution->data.m_points[input_point]);
            }
        }
        BasePreprocessing::forward();
        LOGGING << "Finish 'ignore points'. Output problem: " << SUMMARY(m_output_solution->data) << std::endl;
    }

    void IgnorePoints::backward() {
        BasePreprocessing::backward();
    }
}