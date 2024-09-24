#include <gtest/gtest.h>
#include <honeybadger/honeybadger.cuh>
#include <honeybadger/testlib/testlib.cuh>

#define declare_assert_equal_function_on_arr(name, type)               \
void name(const std::vector<type>& expect) {                           \
    for(size_t i = 0; i < expect.size(); i++) {                        \
        ASSERT_EQ(expect[i], m_solution->data.name[i]) << #name << " " << i; \
    }                                                                  \
}

#define declare_assert_equal_function_on_scalar(name) \
template <typename T>                                 \
void name(const T& expect) {                          \
    ASSERT_EQ(*m_solution->data.name, expect) << #name;     \
}

namespace honeybadger::operations {
    class TestFindBestPointToInsert : public ::testing::Test {
    public:
        void setup(Graph::Ptr graph, const std::vector<Point> &points) {
            m_graph = graph;
            m_problem = std::make_shared<Problem>(m_graph, points);
            m_solution = std::make_shared<Solution>(m_problem);
        }

        void clean() {
            m_graph = nullptr;
            m_problem = nullptr;
            m_solution = nullptr;
        }

        declare_assert_equal_function_on_arr(m_available_points_indices, int_t)

        declare_assert_equal_function_on_scalar(m_num_available_points)

        declare_assert_equal_function_on_arr(m_assigned_points_indices, int_t)

        declare_assert_equal_function_on_scalar(m_num_assigned_points)

        declare_assert_equal_function_on_arr(m_available_points_cross_assigned_points, int_t)

        declare_assert_equal_function_on_arr(m_possible_points_indices, int_t)

        declare_assert_equal_function_on_scalar(m_num_possible_points)

        declare_assert_equal_function_on_arr(m_possible_points_crossing_number, int_t)

        declare_assert_equal_function_on_arr(m_best_points_indices, int_t)

        declare_assert_equal_function_on_scalar(m_num_best_points)

        Graph::Ptr m_graph;
        Problem::Ptr m_problem;
        Solution::Ptr m_solution;
    };

    TEST_F(TestFindBestPointToInsert, test1) {
        // Initialize variables
        // Points will look like following
        // 1   4
        //   2
        // 0   3
        //
        // And the complete graph K4 will be partially assigned as following
        // *   *
        //   1
        // 0   2
        this->setup(deterministic::completeGraph(4), deterministic::unitGrid5With1Center());
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // First test on empty operations
        m_num_available_points(0);
        m_num_assigned_points(0);
        m_num_possible_points(0);
        m_num_best_points(0);

        // Test 1: Find best points on empty operations
        // 1   4
        //   2
        // 0   3
        //
        // *   *
        //   *
        // *   *
        m_solution->findBestPoint(0);
        cudaDeviceSynchronize();

        m_num_available_points(5);
        m_num_assigned_points(0);
        m_num_possible_points(5);
        m_num_best_points(5);

        m_available_points_indices({0, 1, 2, 3, 4});
        m_assigned_points_indices({});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0});
        m_possible_points_indices({0, 1, 2, 3, 4});
        m_possible_points_crossing_number({0, 0, 0, 0, 0});
        m_best_points_indices({0, 1, 2, 3, 4});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 2: Insert first point
        // 1   4
        //   2
        // 0   3
        //
        // *   *
        //   *
        // 0   *
        m_solution->insert(0, 0);
        cudaDeviceSynchronize();
        m_solution->findBestPoint(1);
        cudaDeviceSynchronize();

        m_num_available_points(4);
        m_num_assigned_points(1);
        m_num_possible_points(4);
        m_num_best_points(4);

        m_available_points_indices({1, 2, 3, 4});
        m_assigned_points_indices({0});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0});
        m_possible_points_indices({1, 2, 3, 4});
        m_possible_points_crossing_number({0, 0, 0, 0, 0});
        m_best_points_indices({1, 2, 3, 4});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 2: Insert second point into the center
        // 1   4
        //   2
        // 0   3
        //
        // *   *
        //   1
        // 0   *
        m_solution->insert(1, 2);
        cudaDeviceSynchronize();
        m_solution->findBestPoint(2);
        cudaDeviceSynchronize();

        m_num_available_points(3);
        m_num_assigned_points(2);
        m_num_possible_points(2); // Only 2 of 3 available points are possible now
        m_num_best_points(2);

        m_available_points_indices({1, 3, 4});
        m_assigned_points_indices({0, 2});
        m_available_points_cross_assigned_points({0, 0, 1, 0, 0}); // Inserting 2 into point 4 would cross point 2
        m_possible_points_indices({1, 3});
        m_possible_points_crossing_number({0, 0, 0, 0, 0});
        m_best_points_indices({1, 3});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 3: Insert third point
        // 1   4
        //   2
        // 0   3
        //
        // *   *
        //   1
        // 0   2
        m_solution->insert(2, 3);
        m_solution->findBestPoint(3);
        cudaDeviceSynchronize();

        m_num_available_points(2);
        m_num_assigned_points(3);
        m_num_possible_points(0); // Schachmatt
        m_num_best_points(0);
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Empty operations, every point should be a best point
        this->clean();
    }

    TEST_F(TestFindBestPointToInsert, test2) {
        // Initialize variables
        // Points will look like following
        //    1
        // 0  2  4  5
        //    3
        //
        // And the path graph P4 will be assigned as
        //    0
        // *  *  2  3
        //    1
        this->setup(deterministic::pathGraph(4), deterministic::crossAlikeGrid());

        // Test 1: Find best points on empty operations
        m_solution->findBestPoint(0);
        cudaDeviceSynchronize();

        m_num_available_points(6);
        m_num_assigned_points(0);
        m_num_possible_points(6);
        m_num_best_points(6);

        m_available_points_indices({0, 1, 2, 3, 4, 5});
        m_assigned_points_indices({});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0, 0});
        m_possible_points_indices({0, 1, 2, 3, 4, 5});
        m_possible_points_crossing_number({0, 0, 0, 0, 0, 0});
        m_best_points_indices({0, 1, 2, 3, 4, 5});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 2: Insert point 0 to 1
        //    1
        // 0  2  4  5
        //    3
        //
        //    0
        // *  *  *  *
        //    *
        m_solution->insert(0, 1);
        m_solution->findBestPoint(1);
        cudaDeviceSynchronize();

        m_num_available_points(5);
        m_num_assigned_points(1);
        m_num_possible_points(5);
        m_num_best_points(5);

        m_available_points_indices({0, 2, 3, 4, 5});
        m_assigned_points_indices({1});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0, 0});
        m_possible_points_indices({0, 2, 3, 4, 5});
        m_possible_points_crossing_number({0, 0, 0, 0, 0, 0});
        m_best_points_indices({0, 2, 3, 4, 5});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 3: Insert point 1 to 3
        //    1
        // 0  2  4  5
        //    3
        //
        //    0
        // *  x  *  *
        //    1
        m_solution->insert(1, 3);
        m_solution->findBestPoint(2);
        cudaDeviceSynchronize();

        m_num_available_points(3);
        m_num_assigned_points(2);
        m_num_possible_points(3);
        m_num_best_points(3);

        m_available_points_indices({0, 4, 5});
        m_assigned_points_indices({1, 3});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0, 0});
        m_possible_points_indices({0, 4, 5});
        m_possible_points_crossing_number({0, 0, 0, 0, 0, 0});
        m_best_points_indices({0, 4, 5});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 4: Insert point 2 to 4
        //    1
        // 0  2  4  5
        //    3
        //
        //    0
        // *  x  2  *
        //    1
        m_solution->insert(2, 4);
        m_solution->findBestPoint(3);
        cudaDeviceSynchronize();

        m_num_available_points(2);
        m_num_assigned_points(3);
        m_num_possible_points(2);
        m_num_best_points(1);

        m_available_points_indices({0, 5});
        m_assigned_points_indices({1, 3, 4});
        m_available_points_cross_assigned_points({0, 0, 0, 0, 0, 0});
        m_possible_points_indices({0, 5});
        m_possible_points_crossing_number({1, 0, 0, 0, 0, 0});
        m_best_points_indices({5});
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_FALSE(*m_solution->data.m_is_complete);

        // Test 4: Insert point 3 to 5
        //    1
        // 0  2  4  5
        //    3
        //
        //    0
        // *  x  2  3
        //    1
        m_solution->insert(3, 5);
        ASSERT_TRUE(*m_solution->data.m_is_correct);
        ASSERT_TRUE(*m_solution->data.m_is_complete);
        clean();
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}