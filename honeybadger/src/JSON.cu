#include "honeybadger/JSON.cuh"
#include <nlohmann/json.hpp>
#include <utility>

namespace honeybadger::json {
    using json = nlohmann::json;

    honeybadger::Graph::Ptr readGraph(const std::string &challenge) {
        std::vector<honeybadger::Edge> edges;
        const auto &json_object = json::parse(challenge);
        for (const auto &json_edge: json_object["edges"]) {
            edges.emplace_back(json_edge["source"], json_edge["target"]);
        }
        return std::make_shared<Graph>(edges, static_cast<int_t>(json_object["nodes"].size()));
    }

    honeybadger::Problem::Ptr readProblem(const std::string &challenge, honeybadger::Graph::Ptr graph) {
        std::vector<honeybadger::Point> points;
        const auto &json_object = json::parse(challenge);
        for (const auto &json_point: json_object["points"]) {
            honeybadger::Point point{json_point["x"], json_point["y"]};
            if (std::find(points.begin(), points.end(), point) != points.end()) {
                LOGGING << "Duplicate point found " << point;
                continue;
            }
            points.push_back(point);
        }
        return std::make_shared<Problem>(std::move(graph), points, json_object["id"]);
    }

    std::vector<honeybadger::Point> readAssignment(const std::string &challenge) {
        std::vector<honeybadger::Point> assignment;
        const auto &json_object = json::parse(challenge);
        for (const auto &json_node: json_object["nodes"]) {
            honeybadger::Point point{json_node["x"], json_node["y"]};
            assignment.push_back(point);
        }
        return assignment;
    }

    std::string writeSolution(honeybadger::Solution::Ptr solution) {
        json object;

        // Serialize edges
        std::vector<json> edges_json;
        for (int_t m = 0; m < solution->data.m; m++) {
            auto &edge = solution->data.m_edges[m];
            json edge_json;
            edge_json["source"] = edge.m_source;
            edge_json["target"] = edge.m_target;
            edges_json.push_back(edge_json);
        }
        object["edges"] = json(edges_json);

        // Serialize id
        object["id"] = solution->data.m_id;

        // Serialize nodes
        std::vector<json> nodes_json;
        for (int_t n = 0; n < solution->data.n; n++) {
            auto node_pos_idx = solution->data.m_assignments[n];
            honeybadger::Point node_pos{-1, -1}; // TODO: throw exception if operations is not complete
            if (node_pos_idx != NON_VALID_ASSIGNMENT) {
                node_pos = solution->data.m_points[node_pos_idx];
            }
            json node_json;
            node_json["id"] = n;
            node_json["x"] = node_pos.m_x;
            node_json["y"] = node_pos.m_y;
            nodes_json.push_back(node_json);
        }
        object["nodes"] = json(nodes_json);

        // Serialize points
        std::vector<json> points_json;
        for (int_t p = 0; p < solution->data.p; p++) {
            honeybadger::Point point = solution->data.m_points[p];
            json point_json;
            point_json["id"] = p;
            point_json["x"] = point.m_x;
            point_json["y"] = point.m_y;
            points_json.push_back(point_json);
        }
        object["points"] = json(points_json);

        return to_string(object);
    }

    honeybadger::Solution::Ptr createSolutionFromJson(const std::string &challenge) {
        auto graph = honeybadger::json::readGraph(challenge);
        auto problem = honeybadger::json::readProblem(challenge, graph);
        return std::make_shared<Solution>(problem);
    }
}