#include <gtest/gtest.h>
#include <honeybadger/honeybadger.cuh>
#include <honeybadger/testlib/testlib.cuh>

namespace honeybadger {
    TEST(TestProblem, testInitializeProblem) {
        for (int j = 0; j < 5; j++) {
            auto graph{deterministic::triangleGraph()};
            auto problem = std::make_shared<Problem>(graph, deterministic::unitGrid4Points());
            ASSERT_EQ(4, problem->p);
        }
    }
    TEST(TestProblem, testEqualityAndStressTest) {
        for (int j = 0; j < 5; j++) {
            auto graph1{deterministic::completeGraph(3)};
            auto graph2{deterministic::completeGraph(4)};

            auto problem1 = std::make_shared<Problem>(graph1, deterministic::unitGrid4Points());
            auto problem2 = std::make_shared<Problem>(graph1, deterministic::unitGrid4Points());
            auto problem3 = std::make_shared<Problem>(graph2, deterministic::unitGrid4Points());
            auto problem4 = std::make_shared<Problem>(graph1, deterministic::unitGrid5With1Center());

            ASSERT_EQ(*problem1, *problem2);

            ASSERT_NE(*problem1, *problem3);
            ASSERT_NE(*problem1, *problem4);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}