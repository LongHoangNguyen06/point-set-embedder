#pragma once

#include <iostream>

namespace honeybadger {
    /**
     * Point class.
     */
    typedef struct Point {
        float_t m_x;
        float_t m_y;

        inline __host__ __device__ bool operator==(const Point &rhs) const {
            return m_x == rhs.m_x &&
                   m_y == rhs.m_y;
        }

        inline __host__ __device__ bool operator!=(const Point &rhs) const {
            return !(rhs == *this);
        }

        inline __host__ __device__ bool operator>(const Point &rhs) const {
            return m_x > rhs.m_x || (m_x == rhs.m_x && m_y > rhs.m_y);
        }

        inline  __host__ __device__ bool operator<(const Point &rhs) const {
            return m_x < rhs.m_x || (m_x == rhs.m_x && m_y < rhs.m_y);
        }

        /**
         * @return true if point is valid.
         */
        inline __host__ __device__ bool is_valid() const {
            return ceil(m_x) == m_x && ceil(m_y) == m_y;
        }
    } Point;

    inline std::ostream &operator<<(std::ostream &os, const Point &point) {
        os << "(" << point.m_x << ", " << point.m_y << ")";
        return os;
    }

    /**
     * Segment class.
     */
    typedef struct Segment {
        Point m_start;
        Point m_end;
    } Segment;

    inline std::ostream &operator<<(std::ostream &os, const Segment &segment) {
        os << segment.m_start << " -> " << segment.m_end;
        return os;
    }
}

namespace honeybadger::geometry {

    /**
     * @return the signed distance between a line segment and a point.
     */
    inline __device__ __host__ float signed_distance(const Segment &segment, const Point &P) {
        float_t a = (segment.m_start.m_x - P.m_x) * (segment.m_end.m_y - segment.m_start.m_y);
        float_t b = (segment.m_start.m_y - P.m_y) * (segment.m_end.m_x - segment.m_start.m_x);
        return a - b;
    }

    /**
     * @return true if segment b's two m_end points are on different side of segment a.
     */
    inline __device__ __host__ bool segment_points_on_different_side(const Segment &segment_a, const Segment &segment_b) {
        float_t dist_b_end_a = signed_distance(segment_a, segment_b.m_end);
        float_t dist_b_start_a = signed_distance(segment_a, segment_b.m_start);

        bool b_end_left_of_a = dist_b_end_a > 0;
        bool b_end_right_of_a = dist_b_end_a < 0;

        bool b_start_right_of_a = dist_b_start_a < 0;
        bool b_start_left_of_a = dist_b_start_a > 0;

        return (b_start_right_of_a && b_end_left_of_a) || (b_start_left_of_a && b_end_right_of_a);
    }

    /**
     * @return true if two segments intersect and not just touch.
     */
    inline __device__ __host__ bool intersect(const Segment &segment_a, const Segment &segment_b) {
        return segment_points_on_different_side(segment_a, segment_b) &&
               segment_points_on_different_side(segment_b, segment_a);
    }

    /**
     * @return true x is between a and b
     */
    inline __device__ __host__ bool between(float_t a, float_t b, float_t x) {
        return (a <= x && x <= b) || (b <= x && x <= a);
    }

    /**
     * @return true if point is inside a bounding box of the segment.
     */
    inline __device__ __host__ bool inside_bounding_box(const Segment &segment_a, const Point &point) {
        return between(segment_a.m_start.m_x, segment_a.m_end.m_x, point.m_x) &&
               between(segment_a.m_start.m_y, segment_a.m_end.m_y, point.m_y);
    }

    /**
     * @return true if a segment crosses a point.
     */
    inline __device__ __host__ bool cross(const Segment &segment_a, const Point &point) {
        return inside_bounding_box(segment_a, point) && \
                    abs(signed_distance(segment_a, point)) < 0.000000001 && \
                   segment_a.m_start != point && segment_a.m_end != point;
    }
}