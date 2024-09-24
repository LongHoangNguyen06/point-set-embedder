#pragma once

#include <random>
#include <vector>
#include <honeybadger/honeybadger.cuh>

namespace honeybadger {
    namespace deterministic {
        /**
         * @return Triangle graph consists of three points and three edges.
         */
        Graph::Ptr triangleGraph();

        /**
         * @return A simple path consists of n vertices.
         */
        Graph::Ptr pathGraph(int_t n);

        /**
         * Return following grid on x axis
         * 0 1 .... p
         */
        std::vector<Point> pathPoint(int_t p);

        /**
         * Return following grid
         *    1
         * 0  2  4  5
         *    3
         */
        std::vector<Point> crossAlikeGrid();

        /**
         * @return Integer unit grid consists of 4 points.
         */
        std::vector<Point> unitGrid4Points();

        /**
         * @return Integer unit grid consists of 5 points with 1 center as last.
         */
        std::vector<Point> unitGrid5With1Center();

        /**
         * @return Complete graph of n nodes.
         */
        Graph::Ptr completeGraph(int_t n);

        /**
         * @return Complete integer grid of nxn points.
         */
        std::vector<Point> gridNxN(int_t n);
    }

    namespace random {
        /**
         * Generate random graph
         * @param n nodes
         * @param m edges
         * @return random graph of n nodes and m edges
         */
        Graph::Ptr randomGraph(int_t n, int_t m, int_t seed = 0);

        /**
         * Generate random points in the nxn grid.
         * @param n size of grid
         * @param p number of points
         * @return random grid of p points in nxn grid.
         */
        std::vector<Point> randomPoints(int_t n, int_t p, int_t seed = 0);
    }
}