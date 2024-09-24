#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import json
import numpy as np
from multiprocessing import Pool


def read_problem(content: str) -> nx.Graph:
    challenge = json.loads(content)
    graph = nx.Graph()
    # Read id
    graph.id = challenge["id"]

    # Read node
    for node in challenge["nodes"]:
        graph.add_node(node["id"])

    # Read edge
    for edge in challenge["edges"]:
        graph.add_edge(edge["source"], edge["target"])

    # Read points
    graph.points = []
    for point in challenge["points"]:
        graph.points.append([point["x"], point["y"]])

    # Read assignments
    graph.assignments = []
    for _ in challenge["nodes"]:
        graph.assignments.append(0)

    # Convert to numpy array
    graph.points = np.asarray(graph.points)
    graph.assignments = np.asarray(graph.assignments)

    return graph


def plot(INPUT_PATH):
    FONT_SIZE=75
    INPUT_FILE = os.path.basename(INPUT_PATH)
    OUTPUT_PATH = f"out/problems/{Path(INPUT_FILE).stem}.svg"

    if not os.path.exists(OUTPUT_PATH):
        with open(INPUT_PATH) as f:
            graph = read_problem(f.read())

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))

        # Draw points
        axes[0].scatter([p[0] for p in graph.points], [p[1] for p in graph.points])
        for i, p in enumerate(graph.points):
            axes[0].annotate(i, (p[0], p[1]), fontsize=FONT_SIZE//6)
        axes[0].set_title("Point Set", fontsize=FONT_SIZE)
        axes[0].set_axis_off()
        axes[0].set_title("Point Set", fontsize=FONT_SIZE)
        axes[0].axis('off')# Optional: Remove ticks if still present
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Draw graph
        spring_layout = nx.spring_layout(graph, seed=0)
        nx.draw(graph, ax=axes[1], with_labels=True, pos=spring_layout, node_color="tab:red")
        plt.savefig(OUTPUT_PATH)
        axes[1].set_title("Graph", fontsize=FONT_SIZE)

        plt.savefig(OUTPUT_PATH, bbox_inches='tight')


if __name__ == '__main__':
    os.system("mkdir -p out/problems")
    with Pool(24) as p:
        p.map(plot, glob.glob("problems/*.json"))

