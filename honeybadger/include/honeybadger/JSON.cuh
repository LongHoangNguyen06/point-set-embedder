#pragma once

#include "honeybadger/datastructure/Graph.cuh"
#include "honeybadger/datastructure/Problem.cuh"
#include "honeybadger/datastructure/Solution.cuh"
#include <string>
#include <optional>

namespace honeybadger::json {
    /**
     * Read the graph from the JSON challenge
     * @param challenge input JSON string.
     */
    honeybadger::Graph::Ptr readGraph(const std::string &challenge);

    /**
     * Read the problem from the JSON challenge
     * @param challenge input JSON string.
     */
    honeybadger::Problem::Ptr readProblem(const std::string &challenge, honeybadger::Graph::Ptr graph);

    /**
     * Read the solution from the JSON challenge
     * @param challenge input JSON string.
     */
    honeybadger::Solution::Ptr createSolutionFromJson(const std::string &challenge);

    /**
     * Read the points from the JSON challenge in the node array.
     * @param challenge input JSON string.
     * @return n points where n = graph's #vertices.
     */
    std::vector<honeybadger::Point> readAssignment(const std::string &challenge);

    /**
     * Write the operations into contest's format.
     */
    std::string writeSolution(honeybadger::Solution::Ptr solution);
}