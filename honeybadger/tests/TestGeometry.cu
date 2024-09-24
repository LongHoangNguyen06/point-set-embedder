#include <gtest/gtest.h>
#include <honeybadger/honeybadger.cuh>
#include <honeybadger/testlib/testlib.cuh>

namespace honeybadger::geometry {
    TEST(TestGeometry, testSignedDistance) {
        Segment segment{{0, 0},
                        {1, 0}};
        ASSERT_EQ(0.5, signed_distance(segment, {0.5, 0.5}));
        ASSERT_EQ(-0.75, signed_distance(segment, {0.5, -0.75}));
        ASSERT_EQ(0, signed_distance(segment, {0.5, 0}));
    }

    TEST(TestGeometry, testSegmentPointsOnDifferentSide) {
        Segment segment_a{{0, 0},
                          {1, 0}};
        Segment segment_b{{0.5, 0.5},
                          {0.5, -0.5}};
        Segment segment_c{{0.5, 0.5},
                          {0.5, 1.0}};
        ASSERT_TRUE(segment_points_on_different_side(segment_a, segment_b));
        ASSERT_TRUE(segment_points_on_different_side(segment_c, segment_a));
        ASSERT_FALSE(segment_points_on_different_side(segment_a, segment_c));
    }

    TEST(TestGeometry, testIntersect) {
        Segment segment_a{{0, 0},
                          {1, 0}};
        Segment segment_b{{0.5, 0.5},
                          {0.5, -0.5}};
        Segment segment_c{{0.5, 0.5},
                          {0.5, 1.0}};
        Segment segment_d{{0.5, 0.5},
                          {0.5, 0.0}};
        ASSERT_TRUE(intersect(segment_a, segment_b));
        ASSERT_FALSE(intersect(segment_c, segment_a));
        ASSERT_FALSE(intersect(segment_a, segment_c));
        ASSERT_FALSE(intersect(segment_a, segment_d));
        ASSERT_FALSE(intersect(segment_d, segment_a));
    }

    TEST(TestGeometry, testBetween) {
        ASSERT_TRUE(between(0, 1, 0.5));
        ASSERT_FALSE(between(0, 1, 2));
    }

    TEST(TestGeometry, testInsideBoundingBox) {
        ASSERT_TRUE(inside_bounding_box({{0.0, 0.0},
                                         {2.0, 2.0}}, {1.0, 1.0}));
        ASSERT_FALSE(inside_bounding_box({{0.0, 0.0},
                                          {2.0, 2.0}}, {3.0, 1.0}));
    }

    TEST(TestGeometry, testCross) {
        Segment segment_a{{0, 0},
                          {1, 0}};
        ASSERT_TRUE(cross(segment_a, {0.5, 0}));
        ASSERT_FALSE(cross(segment_a, {0.5, 1}));
        ASSERT_FALSE(cross(segment_a, {0.0, 0.0}));
        ASSERT_FALSE(cross(segment_a, {1.0, 0.0}));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}