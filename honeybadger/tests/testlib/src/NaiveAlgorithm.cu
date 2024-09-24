#include "honeybadger/testlib/NaiveAlgorithm.cuh"
#include "honeybadger/honeybadger.cuh"

namespace honeybadger::algorithms::naive_algorithm {
    void init(Solution::Ptr solution) {
        for (int_t i = 0; i < solution->data.n; i++) {
            // Find best point to insert vertex
            solution->findBestPoint(i);

            if (*solution->data.m_num_best_points == 0) {
                throw_error("Can not find further point to insert " << i, NotSatisfiableException)
            }

            // Insert the vertex into that point
            solution->insert(i, solution->data.m_best_points_indices[0]);
        }
    }

    void imp(Solution::Ptr solution) {
        for (int_t i = 0; i < solution->data.n; i++) {
            // Remove vertex
            assert(*solution->data.m_is_correct);
            auto before_assignment = solution->data.m_assignments[i];
            auto before = *solution->data.m_objective_value;
            solution->remove(i);

            // Find best points to insert vertex
            auto intermediate = *solution->data.m_objective_value;
            solution->findBestPoint(i);

            if (*solution->data.m_num_best_points == 0) {
                throw_error("Can not find any point to reinsert " << i, NotSatisfiableException)
            }

            // Reinsert vertex into the best new point
            solution->insert(i, solution->data.m_best_points_indices[0]);

            auto after = *solution->data.m_objective_value;
            auto after_assignment = solution->data.m_assignments[i];
            if (after > before) {
                throw_error("Reinserting node " << i << " from point " << before_assignment << " to point "
                                                << after_assignment << " worsened the operations from "
                                                << before << " to " << intermediate << " to " << after
                                                << ". This is not possible and properly a bug!.",
                            std::runtime_error)
            }
        }
    }
}