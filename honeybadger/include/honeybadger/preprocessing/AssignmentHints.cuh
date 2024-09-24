#pragma once

#include <map>
#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "BasePreprocessing.cuh"

namespace honeybadger::preprocessing::assignment_hints {
    /**
     * Apply hints (which vertex shall map to which point) from user.
     */
    class AssignmentHints : public BasePreprocessing {
    protected:
        void transferAssignmentFromInputToOutput() override;

        void transferAssignmentFromOutputToInput() override;

        void produceOutputSolutions() override;

        void produceOutputParameters() override;
    public:
        AssignmentHints(Solution::Ptr solution, const parameters::AlgorithmParameters& parameter);

        void forward() override;

        void backward() override;
    };
}