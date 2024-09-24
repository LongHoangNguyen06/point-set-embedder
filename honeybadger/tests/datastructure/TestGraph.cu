#include <gtest/gtest.h>
#include <honeybadger/honeybadger.cuh>
#include <honeybadger/testlib/testlib.cuh>

namespace honeybadger {
    TEST(TestGraph, testTriangleGraph) {
        for (int j = 0; j < 5; j++) {
            Graph::Ptr G{deterministic::triangleGraph()};
            for (int i = 0; i < 10; i++) {
                ASSERT_EQ(3, G->n);
                ASSERT_EQ(3, G->m);

                ASSERT_EQ(2, G->m_degree[0]);
                ASSERT_EQ(2, G->m_degree[1]);
                ASSERT_EQ(2, G->m_degree[2]);

                ASSERT_EQ(G->m_edges[0], Edge(0, 1));
                ASSERT_EQ(G->m_edges[1], Edge(0, 2));
                ASSERT_EQ(G->m_edges[2], Edge(1, 2));

                ASSERT_EQ(G->m_incidence_matrix[0][0], 0);
                ASSERT_EQ(G->m_incidence_matrix[0][1], 1);
                ASSERT_EQ(G->m_incidence_matrix[1][0], 0);
                ASSERT_EQ(G->m_incidence_matrix[1][1], 2);
                ASSERT_EQ(G->m_incidence_matrix[2][0], 1);
                ASSERT_EQ(G->m_incidence_matrix[2][1], 2);
            }
        }
    }

    TEST(TestGraph, testK4) {
        Graph::Ptr G {deterministic::completeGraph(4)};
        ASSERT_EQ(G->m_edges[0], Edge(0, 1));
        ASSERT_EQ(G->m_edges[1], Edge(0, 2));
        ASSERT_EQ(G->m_edges[2], Edge(0, 3));
        ASSERT_EQ(G->m_edges[3], Edge(1, 2));
        ASSERT_EQ(G->m_edges[4], Edge(1, 3));
        ASSERT_EQ(G->m_edges[5], Edge(2, 3));
    }

    TEST(TestGraph, testEqualityAndStressTest) {
        for (int_t i = 0; i < 10; i++) {
            Graph::Ptr G1{random::randomGraph(500, 1000, i * 1000)};
            Graph::Ptr G2{random::randomGraph(500, 1000, (i + 1) * 1000)};
            ASSERT_NE(*G1, *G2);
        }
        for (int_t i = 10; i < 20; i++) {
            Graph::Ptr G1{deterministic::completeGraph(i)};
            Graph::Ptr G2{deterministic::completeGraph(i)};
            ASSERT_EQ(*G1, *G2);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}