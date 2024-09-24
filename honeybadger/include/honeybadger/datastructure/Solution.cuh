#pragma once

#include "Graph.cuh"
#include "Problem.cuh"
#include <optional>
#include <ostream>

namespace honeybadger {
/**
 * Represent a operations.
 */
typedef struct Solution {
  using Ptr = std::shared_ptr<Solution>;

  typedef struct Data {
    int_t *m_assignments{}; // M[n]. M[i] = position of vertex i. Unassigned
                            // vertex has value NON_VALID_ASSIGNMENT.
    int_t *m_point_assigned_to{}; // M[p]. M[i] = assigned vertex of point i.
                                  // Unassigned point has value
                                  // NON_VALID_ASSIGNMENT.

    /// For computing crossing number
    /// See operations/RecomputeSolution.cuh
    bool **m_edge_cross_edge{};    // M[m, m]
    int_t *m_edge_crossings{};     // M[m]
    int_t *m_objective_value{};    // M[1]
    int_t *m_node_crossings{};     // M[n]
    bool **m_point_crossed_edge{}; // M[p, m]
    int_t *m_point_crossings{};    // M[p]

    /**
     * A solution is correct iff
     * [C1] each assigned vertex v has a valid point p
     * [C1.1] and point p is also assigned to vertex v
     * [C2] if a point p is mapped to vertex v then v must be valid and v must
     * be mapped to p [C3] no assigned point is crossed [C4] no two vertices
     * have the same point
     *
     * A solution is complete iff
     * [C5] every vertex has a valid point
     *
     * A solution is only accepted iff
     * [A1] it's correct AND complete.
     */
    bool *m_is_correct{};  // M[1].
    bool *m_is_complete{}; // M[1].

    /// For computing best points to insert node
    /// See operations/FindBestPoint.cuh
    int_t *m_available_points_indices{}; // M[p]
    int_t *m_num_available_points{};     // M[1]
    int_t *m_assigned_points_indices{};  // M[p]
    int_t *m_num_assigned_points{};      // M[1]

    bool *m_available_points_cross_assigned_points{}; // M[p]
    int_t *m_possible_points_indices{};               // M[p]
    int_t *m_num_possible_points{};                   // M[1]

    int_t *m_possible_points_crossing_number{}; // M[p]
    int_t *m_best_points_indices{};             // M[p]
    int_t *m_num_best_points{};                 // M[1]

    /// Attributes of graph and problem.
    /// Don't call free on those attributes
    int_t n{0};                          // Number of graph's nodes.
    int_t m{0};                          // Number of graph's edges.
    int_t p{0};                          // Cardinality of point set.
    int_t m_id{0};                       // Problem's id.
    Graph* m_graph{};
    Problem* m_problem{};
    Edge *m_edges{nullptr};              // M[m]
    int_t **m_incidence_matrix{nullptr}; // M[n, d...]
    int_t *m_degree{nullptr};            // M[n]
    Point *m_points{};                   // M[p]

    Data(Problem::Ptr problem);

    /**
   * Check if two edges cross.
   *
   * @param edge_idx_a position of edge a.
   * @param edge_idx_b position of edge b.
   * @return true if edge a and edge b cross.
     */
    [[nodiscard]] inline __device__ __host__ bool
    doesEdgeCrossesEdge(int_t edge_idx_a, int_t edge_idx_b) const {
      auto &edge_a = m_edges[edge_idx_a];
      auto &edge_b = m_edges[edge_idx_b];

      auto edge_a_visible = isEdgeVisible(edge_a);
      auto edge_b_visible = isEdgeVisible(edge_b);

      // Return result
      return (edge_idx_a != edge_idx_b) && (edge_a_visible && edge_b_visible) &&
             geometry::intersect(getVisibleSegment(edge_a),
                                 getVisibleSegment(edge_b));
    }

    /**
   * Check if edge cross point.
   *
   * @param edge_idx position of edge.
   * @param point_idx position of point.
   * @return true if edge cross point, i.e., edge is visible, cross point and
   * point does not belong to edge's endpoints.
     */
    [[nodiscard]] inline __device__ __host__ bool
    isPointCrossedByEdge(int_t point_idx, int_t edge_idx) const {
      auto &edge = m_edges[edge_idx];
      auto &point = m_points[point_idx];

      auto edge_visible = isEdgeVisible(edge);

      return edge_visible && geometry::cross(getVisibleSegment(edge), point);
    }

    /**
   * Check if a point is assigned.
     */
    [[nodiscard]] inline __device__ __host__ bool
    isPointAssigned(int_t point) const {
      return m_point_assigned_to[point] != NON_VALID_ASSIGNMENT;
    }

    /**
   * Check if a point is crossed.
     */
    [[nodiscard]] inline __device__ __host__ bool
    isPointCrossed(int_t point) const {
      return m_point_crossings[point] > 0;
    }

    /**
   * Check if a vertex is visible, i.e., has value other than
   * NON_VALID_ASSIGNMENT.
     */
    [[nodiscard]] inline __device__ __host__ bool
    isVertexVisible(int_t vertex) const {
      return m_assignments[vertex] != NON_VALID_ASSIGNMENT;
    }

    /**
   * Check if an edge is visible, yet.
   *
   * @param edge dge
   * @return true if both nodes of edge are visible..
     */
    [[nodiscard]] inline __device__ __host__ bool
    isEdgeVisible(const Edge &edge) const {
      return isVertexVisible(edge.m_source) && isVertexVisible(edge.m_target);
    }

    /**
   * Check if an edge is visible, yet.
   *
   * @param edge_idx position of edge
   * @return true if both nodes of edge are assigned.
     */
    [[nodiscard]] inline __device__ __host__ bool
    isEdgeVisible(int_t edge_idx) const {
      return isEdgeVisible(m_edges[edge_idx]);
    }

    /**
   * Get segment of edge. Assume edge to be visible.
   *
   * @param edge edge
   * @return Segment.
     */
    [[nodiscard]] inline __device__ __host__ Segment
    getVisibleSegment(const Edge &edge) const {
      auto from_point_idx = m_assignments[edge.m_source];
      auto to_point_idx = m_assignments[edge.m_target];
      return Segment{.m_start = m_points[from_point_idx],
                     .m_end = m_points[to_point_idx]};
    }

    /**
   * Get segment of edge. Assume edge to be visible.
   *
   * @param edge_idx edge position
   * @return Segment.
     */
    [[nodiscard]] inline __device__ __host__ Segment
    getVisibleSegment(int_t edge_idx) const {
      auto &edge = m_edges[edge_idx];
      return getVisibleSegment(edge);
    }

    bool operator==(const Data &rhs) const;

    bool operator!=(const Data &rhs) const;

    std::ostream &operator<<(std::ostream &os);
  } Data;
  Graph::Ptr m_graph{};
  Problem::Ptr m_problem{};
  bool m_own_data{false}; // Not own the data by default
  Data data;

  Solution(Problem::Ptr problem);

  Solution(const Solution &solution);

  ~Solution();

  void insert(int_t vertex, int_t point_idx);

  void remove(int_t vertex);

  void update();

  void assign(const std::vector<Point>& points);

  std::vector<int_t> findBestPoint(int_t vertex);

  bool operator==(const Solution &rhs) const;

  bool operator!=(const Solution &rhs) const;

  std::ostream &operator<<(std::ostream &os);
} Solution;
} // namespace honeybadger