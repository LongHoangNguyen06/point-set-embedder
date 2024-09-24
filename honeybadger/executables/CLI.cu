#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include "CLI.cuh"

namespace honeybadger::cli {
    static std::map<std::string, parameters::Mode> modeMap{
            {"init",             parameters::Mode::INITIALIZE},
            {"improve",          parameters::Mode::IMPROVE},
            {"init_and_improve", parameters::Mode::INITIALIZE_AND_IMPROVE}
    };

    const auto IGNORE_EDGES_STRING = "--ie";
    const auto IGNORE_VERTICES_STRING = "--iv";
    const auto IGNORE_POINTS_STRING = "--ip";
    const auto HINT_STRING = "-f,--fix";
     static std::map<std::string, parameters::Preprocessing> preprocessingMap{
            {IGNORE_EDGES_STRING, parameters::Preprocessing::IGNORE_EDGES},
            {HINT_STRING, parameters::Preprocessing::HINTS},
            {IGNORE_VERTICES_STRING, parameters::Preprocessing::IGNORE_VERTICES},
            {IGNORE_POINTS_STRING, parameters::Preprocessing::IGNORE_POINTS},
    };


    void addOptions(CLI::App *app, Parameters &parameter) {
                app->add_option("-i,--input", parameter.m_input_json_file_name,
                        "[Required] Path to input JSON file.")->required()->check(CLI::ExistingFile, "Contest file does not exist.");
        app->add_option("-o,--output", parameter.m_output_json_file_name,
                        "[Optional] Path to output JSON file.");
        app->add_option("-m,--mode", parameter.m_algorithm_parameter.m_mode,
                        "[Optional] Which should be done with the JSON file.\nOption 'init' tries to find a feasible solution.\nOption 'improve' tries to improve the current correct solution in input file.\nOption 'init_and_improve' with find a feasible solution and improve it.")
                        ->transform(CLI::CheckedTransformer(modeMap, CLI::ignore_case))
                        ->default_val("init_and_improve");
        app->add_option("-n,--neighbor", parameter.m_algorithm_parameter.m_neighborhood_size,
                        "[Optional] Size of local search's neighborhood.")
                ->transform(CLI::CheckedTransformer(modeMap, CLI::ignore_case))
                ->default_val("init_and_improve");
    }

    void addPreprocessingOptions(CLI::App *app, Parameters &parameter) {
        app->add_option(IGNORE_EDGES_STRING, parameter.m_algorithm_parameter.m_ignored_edges,
                        "[Optional] Dictates which edges will be ignored by the algorithm..");
        app->add_option(HINT_STRING, parameter.m_algorithm_parameter.m_hints,
                        "[Optional] Dictates which vertex should be assigned to which point. An algorithm has the option to ignore this completely.");
        app->add_option(IGNORE_VERTICES_STRING, parameter.m_algorithm_parameter.m_ignored_vertices,
                        "[Optional] Dictates which vertices should be ignored.");
        app->add_option(IGNORE_POINTS_STRING, parameter.m_algorithm_parameter.m_ignored_points,
                        "[Optional] Dictates which points should be ignored.");
    }

    /**
     * Find out in which order user wants the pre-processing to happens.
     *
     * @param parameter write ordering into this parameter
     */
    void parsePreprocessingOption(Parameters & parameter, int argc, char ** argv) {
        for(int i = 0; i < argc; i++) {
            const std::string arg = argv[i];
            if (preprocessingMap.find(arg) != preprocessingMap.end()) {
                parameter.m_algorithm_parameter.m_preprocessing_order.push_back(preprocessingMap[arg]);
            }
        }
    }

    /**
     * @param parameter run parameter of the program.
     * @return the json contest string.
     */
    std::string readChallenge(const std::string& file_name) {
        std::stringstream buffer;
        const std::ifstream f(file_name);
        buffer << f.rdbuf();
        return buffer.str();
    }

    Parameters parseCliParameter(int argc, char **argv) {
        Parameters parameter;
        CLI::App app{"HoneyBadger"};
        app.require_subcommand(1, 1);

        CLI::App *random = app.add_subcommand("random", "Random search solver. This solver is the best for random structure.");
        CLI::App *nearest = app.add_subcommand("nearest", "Nearest neighbor solver. Find position that is nearest to all neighbors' positions. This solver is the best for very structured graph. Best in combination with options --hints.");

        // Declare general options
        addOptions(random, parameter);
        addOptions(nearest, parameter);

        // Declare preprocessing options
        addPreprocessingOptions(random, parameter);
        addPreprocessingOptions(nearest, parameter);

        // Parse options
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            std::exit(app.exit(e));
        }

        // See which algorithm was chosen
        if (random->parsed()) {
            parameter.m_algorithm_parameter.m_algorithm = parameters::RANDOM;
        }
        if (nearest->parsed()) {
            parameter.m_algorithm_parameter.m_algorithm = parameters::NEAREST_NEIGHBOR;
        }

        // Parse options order
        parsePreprocessingOption(parameter, argc, argv);

        // Read input string
        parameter.m_json = readChallenge(parameter.m_input_json_file_name);

        return parameter;
    }
}