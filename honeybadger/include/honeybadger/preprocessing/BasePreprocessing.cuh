#pragma once

#include "honeybadger/Utils.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include "honeybadger/Parameters.cuh"
#include <vector>
#include <memory>

namespace honeybadger::preprocessing {
    class BasePreprocessing {
    protected:
        Solution::Ptr m_input_solution;
        parameters::AlgorithmParameters m_input_parameter;

        virtual void transferAssignmentFromInputToOutput() = 0;

        virtual void transferAssignmentFromOutputToInput() = 0;

        virtual void produceOutputSolutions() = 0;

        virtual void produceOutputParameters() = 0;
    public:
        using Ptr = std::shared_ptr<BasePreprocessing>;
        std::vector<Solution::Ptr> m_output_solutions;
        std::vector<parameters::AlgorithmParameters> m_output_parameters;

        explicit BasePreprocessing(Solution::Ptr input_solution,
                                   const parameters::AlgorithmParameters& parameter)
                                   : m_input_solution(input_solution), m_input_parameter(parameter){}

        virtual void forward() {
            this->produceOutputSolutions();
            this->produceOutputParameters();
            this->transferAssignmentFromInputToOutput();
        };

        virtual void backward() {
            this->transferAssignmentFromOutputToInput();
        };
    };
}