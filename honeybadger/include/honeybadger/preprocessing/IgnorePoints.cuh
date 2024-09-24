#pragma once

#include <map>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "BasePreprocessing.cuh"

namespace honeybadger::preprocessing::ignore_points {
    /**
     * Delete points in the input point set.
     */
    class IgnorePoints : public BasePreprocessing {
    private:
        std::map<int_t, int_t> m_input_points_to_output_points;
        std::map<int_t, int_t> m_output_points_to_input_points;
        std::vector<Point> m_kept_points;
        Problem::Ptr m_output_problem{};
        Solution::Ptr m_output_solution{};
        parameters::AlgorithmParameters m_output_parameter;
    protected:
        void transferAssignmentFromInputToOutput() override;

        void transferAssignmentFromOutputToInput() override;

        void produceOutputSolutions() override;

        void produceOutputParameters() override;
    public:
        IgnorePoints(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter);

        void forward() override;

        void backward() override;
    };
}