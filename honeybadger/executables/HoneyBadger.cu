#include <fstream>
#include <filesystem>
#include <honeybadger/honeybadger.cuh>
#include "CLI.cuh"

constexpr auto LOGO = "                ___,,___\n"
                      "           _,-='=- =-  -`\"--.__,,.._\n"
                      "        ,-;// /  - -       -   -= - \"=.\n"
                      "      ,'///    -     -   -   =  - ==-=\\`.\n"
                      "     |/// /  =    `. - =   == - =.=_,,._ `=/|\n"
                      "    ///    -   -    \\  - - = ,ndDMHHMM/\\b  \\\\\n"
                      "  ,' - / /        / /\\ =  - /MM(,,._`YQMML  `|\n"
                      " <_,=^Kkm / / / / ///H|wnWWdMKKK#\"\"-;. `\"0\\  |\n"
                      "        `\"\"QkmmmmmnWMMM\\\"\"WHMKKMM\\   `--. \\> \\\n"
                      " hjm          `\"\"'  `->>>    ``WHMb,.    `-_<@)\n"
                      "                                `\"QMM`.\n"
                      "                                   `>>>";

/**
 * Get output path of a file by append objective value at the end of the stem.
 *
 * @param userOutputPath path where user wants it to be.
 * @param objective_value objective value of the path
 * @return a string path
 */
std::string getOutputPath(const std::string &userOutputPath, int_t objective_value) {
    namespace fs = std::filesystem;

    fs::path p = userOutputPath;
    auto parent_path = p.parent_path();
    parent_path = parent_path.empty() ? "out" : parent_path;
    if (!fs::exists(parent_path)) {
        fs::create_directory(parent_path);
    }
    auto stem = p.stem();
    auto ext = p.extension();
    return parent_path.string() + "/" + stem.string() + "_" + std::to_string(objective_value) + ext.string();
}

/**
 * Get file name of path
 *
 * @param userInputPath path of input file
 * @return a string path
 */
std::string getFileName(const std::string &userInputPath) {
    namespace fs = std::filesystem;
    fs::path p = userInputPath;
    return p.filename();
}

int main(int argc, char **argv) {
    // Parse input
    honeybadger::cli::Parameters parameter = honeybadger::cli::parseCliParameter(argc, argv);

    // Initialize solution
    auto solution = honeybadger::json::createSolutionFromJson(parameter.m_json);

    // Print swag
    std::cerr << LOGO << std::endl;
    LOGGING << "Start solving the challenge '" << parameter.m_input_json_file_name << "'" << std::endl;

    // Start solving
    honeybadger::solver_tree::SolverTree solverTree(parameter.m_algorithm_parameter, solution, parameter.m_json);
    try {
        solverTree.solve();
    } catch (honeybadger::NotSatisfiableException & e) {
        LOGGING_NL << e.what() << std::endl;
        std::exit(1);
    }

    // Check if solution is correct
    if (!*solution->data.m_is_correct) {
        LOGGING_NL << "Found solution is not correct. This is properly a bug" << std::endl;
        std::exit(1);
    }

    // Check if solution is complete
    if (!*solution->data.m_is_complete) {
        LOGGING_NL << "Found solution is not complete. This is properly a bug" << std::endl;
        std::exit(1);
    }

    // Output the solution to file
    auto output_path = getOutputPath(parameter.m_output_json_file_name.has_value() ? parameter.m_output_json_file_name.value() : getFileName(parameter.m_input_json_file_name),
                                     *solution->data.m_objective_value);
    LOGGING_NL << "Solution has crossing number " << *solution->data.m_objective_value << ". Saved at " << output_path << std::endl;
    std::ofstream output_file(output_path);
    output_file << honeybadger::json::writeSolution(solution);
    output_file.close();
}