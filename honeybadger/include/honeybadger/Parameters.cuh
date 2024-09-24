#pragma once
#include "Utils.cuh"
#include <map>
#include <string>
#include <utility>
#include <memory>
namespace honeybadger::parameters {
    typedef enum Mode {
        INITIALIZE, // Just find a feasible solution
        IMPROVE, // Given a feasible solution as input, improve it
        INITIALIZE_AND_IMPROVE // Find a feasible solution and improve it
    } Mode;

    typedef enum Algorithm {
        RANDOM, // Local search
        NEAREST_NEIGHBOR // Nearest neighbor search, work best with partial solution.
    } Algorithm;

    typedef enum Preprocessing {
        HINTS, // Hints
        IGNORE_EDGES, // Ignore edges
        IGNORE_VERTICES, // Ignore vertices
        IGNORE_POINTS, // Ignore points
    } Preprocessing;

    typedef struct AlgorithmParameters {
        std::vector<Preprocessing> m_preprocessing_order{}; // In which order the preprocessing should happen.
        std::set<int_t> m_ignored_vertices{}; // Those vertices should be ignored.
        std::set<std::pair<int_t, int_t>> m_ignored_edges{}; // Those edges should not be used.
        std::set<int_t> m_ignored_points{}; // Those points should be ignored.
        std::map<int_t, int_t> m_hints{}; // Hints of user which vertex should be mapped to which point.
        std::optional<int_t> m_neighborhood_size{}; // Size of neighborhood for random search
        Mode m_mode{INITIALIZE_AND_IMPROVE};
        Algorithm m_algorithm{RANDOM};
    } AlgorithmParameters;
}