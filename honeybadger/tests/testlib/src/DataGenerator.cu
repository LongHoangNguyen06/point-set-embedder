#include <vector>
#include "honeybadger/testlib/DataGenerator.cuh"

namespace honeybadger {
    namespace deterministic {
        Graph::Ptr triangleGraph() {
            return std::make_shared<Graph>(std::vector<Edge>{{0, 1},
                               {0, 2},
                               {1, 2}}, 3);
        }

        Graph::Ptr pathGraph(int_t n) {
            std::vector<Edge> edges;
            for (int_t i = 0; i < n - 1; i++) {
                edges.emplace_back(i, i + 1);
            }
            return std::make_shared<Graph>(edges, n);
        }

        std::vector<Point> pathPoint(int_t p) {
            std::vector<Point> points;
            for (int_t i = 0; i < p; i++) {
                points.push_back(Point{0.0f, float(i)});
            }
            return points;
        }

        Graph::Ptr completeGraph(int_t n) {
            std::vector<Edge> edges;
            for (int_t i = 0; i < n; i++) {
                for (int_t j = i + 1; j < n; j++) {
                    edges.emplace_back(i, j);
                }
            }
            return std::make_shared<Graph>(edges, n);
        }

        std::vector<Point> unitGrid4Points() {
            return std::vector<Point>{{0.0, 0.0},
                                      {0.0, 1.0},
                                      {1.0, 0.0},
                                      {1.0, 1.0}};
        }

        std::vector<Point> unitGrid5With1Center() {
            return std::vector<Point>{{0.0, 0.0},
                                      {0.0, 2.0},
                                      {1.0, 1.0},
                                      {2.0, 0.0},
                                      {2.0, 2.0}};
        }

        std::vector<Point> gridNxN(int_t n) {
            std::vector<Point> ret;
            for (int_t i = 0; i < n; i++) {
                for (int_t j = 0; j < n; j++) {
                    ret.push_back({static_cast<float>(i), static_cast<float>(j)});
                }
            }
            return ret;
        }

        std::vector<Point> crossAlikeGrid() {
            return std::vector<Point>({{0, 0},
                                       {1, 1},
                                       {1, 0},
                                       {1, -1},
                                       {2, 0},
                                       {3, 0}
                                      });
        }
    }

    namespace random {
        Graph::Ptr randomGraph(int_t n, int_t m, int_t seed) {
            std::vector<Edge> edges;
            for (int_t i = 0; i < n; i++) {
                for (int_t j = i + 1; j < n; j++) {
                    edges.emplace_back(i, j);
                }
            }
            assert(m <= edges.size());
            std::shuffle(edges.begin(), edges.end(), std::default_random_engine(seed));
            return std::make_shared<Graph>(std::vector<Edge>(edges.begin(), edges.begin() + m), n);
        }

        std::vector<Point> randomPoints(int_t n, int_t p, int_t seed) {
            std::vector<Point> all = deterministic::gridNxN(n);
            assert(p <= all.size());
            std::shuffle(all.begin(), all.end(), std::default_random_engine(seed));
            return {all.begin(), all.begin() + p};
        }
    }
}