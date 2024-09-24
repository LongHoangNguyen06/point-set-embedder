#pragma once

#include <optional>
#include <string>
#include <map>
#include <set>
#include <honeybadger/honeybadger.cuh>

namespace honeybadger::cli {
    typedef struct Parameters {
        std::string m_input_json_file_name; // Path to input file.
        std::string m_json; // Content of input file.
        std::optional<std::string> m_output_json_file_name; // Path to output file.
        parameters::AlgorithmParameters m_algorithm_parameter; // Parsed content of user's CLI.
    } Parameters;

    /**
     * Parse user arguments into solver's parameters.
     */
    Parameters parseCliParameter(int argc, char **argv);
}