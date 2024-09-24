#include <utility>

#include "honeybadger/preprocessing/AssignmentHints.cuh"

namespace honeybadger::preprocessing::assignment_hints {
    AssignmentHints::AssignmentHints(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter)
    : BasePreprocessing(std::move(solution), parameter){}

    void AssignmentHints::forward() {
        LOGGING << "Start 'assignment hints'. Input problem: " << SUMMARY(m_input_solution->data) << std::endl;
        // Step 1: Free space for the hints
        for (auto hint: m_input_parameter.m_hints) {
            // If vertex of hint is assigned, remove it first
            if (m_input_solution->data.isVertexVisible(hint.first)) {
                m_input_solution->remove(hint.first);
            }
            // If point of hint is assigned, free the point first
            if (m_input_solution->data.isPointAssigned(hint.second)) {
                m_input_solution->remove(m_input_solution->data.m_point_assigned_to[hint.second]);
            }
        }

        // Step 2: Reinsert the vertices in the points of hints.
        for (auto hint: m_input_parameter.m_hints) {
            LOGGING << "Assign vertex " << hint.first << " to point " << hint.second << std::endl;
            m_input_solution->insert(hint.first, hint.second);
        }

        BasePreprocessing::forward();
        LOGGING << "Finish 'assignment hints'. Output problem: " << SUMMARY(m_output_solutions.front()->data) << std::endl;
    }

    void AssignmentHints::backward() {
        BasePreprocessing::backward();
        LOGGING << "Propagate from 'assignment hints' to parent" << std::endl;
    }

    void AssignmentHints::transferAssignmentFromInputToOutput() {

    }

    void AssignmentHints::transferAssignmentFromOutputToInput() {

    }

    void AssignmentHints::produceOutputSolutions() {
        this->m_output_solutions.push_back(m_input_solution);
    }

    void AssignmentHints::produceOutputParameters() {
        this->m_output_parameters.emplace_back(m_input_parameter);
    }
}
