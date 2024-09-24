#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <honeybadger/honeybadger.cuh>
#include <honeybadger/testlib/testlib.cuh>

using namespace std;

namespace fs = std::filesystem;

/**
 * https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
 * Vector of substrings of the string.
 * @return
 */
vector<string> split(const string &s, const string &delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

/**
 * List every file in a directory.
 * @param dirName directory.
 * @param paths output paths.
 * @param objective_values output objective values of the path.
 */
void filesIn(const std::string &dirName,
             std::vector<std::string> &paths,
             std::vector<std::vector<int_t>> &objective_values) {

    std::vector<fs::path> orig_paths;
    for (const auto &entry: fs::directory_iterator(dirName)) {
        orig_paths.push_back(entry.path());
    }

    std::sort(orig_paths.begin(), orig_paths.end());

    for (const auto &path: orig_paths) {
        // Get path names
        paths.push_back(path.string());
        // Get objective value
        const auto &stem = path.filename().stem().string();
        const auto &delimited = split(stem, "_");
        std::vector<int_t> challenge_obj;
        for (int_t i = 1; i < delimited.size(); i++) {
            challenge_obj.push_back(std::stoi(delimited[i]));
        }
        objective_values.push_back(challenge_obj);
    }
}

/**
 * Read whole file into a string and return it
 */
std::string read(const std::string &filename) {
    std::ifstream f(filename);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

namespace honeybadger {
    TEST(TestIntegration, testCr) {
        std::vector<std::string> paths;
        std::vector<std::vector<int_t>> objective_values;
        filesIn("../data/test_cr", paths, objective_values);

#pragma omp parallel for num_threads(48)
        for (int_t i = 0; i < paths.size(); i++) {
            auto challenge = read(paths[i]);
            auto solution = honeybadger::json::createSolutionFromJson(challenge);
            auto assignment = honeybadger::json::readAssignment(challenge);
            solution->assign(assignment);
            ASSERT_EQ(*solution->data.m_objective_value, objective_values[i][0])
                                        << "Failing at file '" << paths[i] << "'." << std::endl;

            ASSERT_EQ(*solution->data.m_is_correct, true);
            ASSERT_EQ(*solution->data.m_is_complete, true);
            std::cout << "Testing cr at file '" << paths[i] << "' was successful." << std::endl;
        }
    }

    TEST(TestIntegration, testInit) {
        std::vector<std::string> paths;
        std::vector<std::vector<int_t>> objective_values;
        filesIn("../data/test_init", paths, objective_values);

#pragma omp parallel for num_threads(48)
        for (int_t i = 0; i < paths.size(); i++) {
            auto challenge = read(paths[i]);
            auto solution = honeybadger::json::createSolutionFromJson(challenge);
            honeybadger::algorithms::naive_algorithm::init(solution);
            ASSERT_EQ(*solution->data.m_objective_value, objective_values[i][0])
                                        << "Failing at file '" << paths[i] << "'." << std::endl;
            ASSERT_EQ(*solution->data.m_is_correct, true);
            ASSERT_EQ(*solution->data.m_is_complete, true);
            std::cout << "Testing init at file '" << paths[i] << "' was successful." << std::endl;
        }
    }

    TEST(TestIntegration, testImp) {
        std::vector<std::string> truth_paths;
        std::vector<std::vector<int_t>> truth_objective_values;
        filesIn("../data/test_imp", truth_paths, truth_objective_values);

        std::vector<std::string> input_paths;
        std::vector<std::vector<int_t>> input_objective_values;
        filesIn("../data/test_init", input_paths, input_objective_values);

#pragma omp parallel for num_threads(48)
       for (int_t i = 0; i < truth_paths.size(); i++) {
            auto challenge = read(input_paths[i]);
            auto solution = honeybadger::json::createSolutionFromJson(challenge);
            auto assignment = honeybadger::json::readAssignment(challenge);
           solution->assign(assignment);
            ASSERT_EQ(*solution->data.m_objective_value, truth_objective_values[i][0])
                                        << "Failing at file '" << truth_paths[i] << "' and '" << input_paths[i] << "'." << std::endl;
            honeybadger::algorithms::naive_algorithm::imp(solution);
            ASSERT_EQ(*solution->data.m_objective_value, truth_objective_values[i][1])
                                        << "Failing at file '" << truth_paths[i] << "' and '" << input_paths[i] << "'." << std::endl;
           ASSERT_EQ(*solution->data.m_is_correct, true);
           ASSERT_EQ(*solution->data.m_is_complete, true);
            std::cout << "Testing imp at file " << truth_paths[i] << " was successful." << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}