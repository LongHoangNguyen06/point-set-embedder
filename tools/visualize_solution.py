#!/usr/bin/env python3

import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
from multiprocessing import Pool


def read_solution(content) -> nx.Graph:
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
        graph.points.append((point["x"], point["y"]))

    # Read assignments
    graph.assignments = []
    for node in challenge["nodes"]:
        point = (node["x"], node["y"])
        graph.assignments.append(graph.points.index(point))

    # Convert to numpy array
    graph.points = np.asarray(graph.points)
    graph.assignments = np.asarray(graph.assignments)

    return graph


def plot(INPUT_PATH):
    FONT_SIZE=90
    INPUT_FILE = os.path.basename(INPUT_PATH)
    OUTPUT_PATH = f"out/solutions/{Path(INPUT_FILE).stem}.svg"

    if not os.path.exists(OUTPUT_PATH):
        print(INPUT_FILE)
        with open(INPUT_PATH) as f:
            graph = read_solution(f.read())
        positions = graph.points[graph.assignments]
        positions = dict(zip(graph, positions))
        cr = INPUT_FILE.split(".")[0].split("_")[-1]

        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))

        # Draw points
        axes[0].scatter([p[0] for p in graph.points], [p[1] for p in graph.points], marker="D")
        for i, p in enumerate(graph.points):
            axes[0].annotate(i, (p[0], p[1]))
        axes[0].set_axis_off()
        axes[0].set_title("Point set", fontsize=FONT_SIZE)
        axes[0].axis('off')# Optional: Remove ticks if still present
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Optionally remove margins if needed
        axes[0].margins(0)

        # Ensure nothing else is plotted
        axes[0].set_axis_off()

        # Draw graph
        spring_layout = nx.spring_layout(graph, seed=0)
        nx.draw(graph, ax=axes[1], with_labels=True, node_color="tab:red", pos=spring_layout)
        axes[1].set_title("Graph", fontsize=FONT_SIZE)

        # Draw solution
        nx.draw(graph, pos=positions, ax=axes[2], with_labels=True, node_color="tab:red")
        axes[2].scatter([p[0] for p in graph.points], [p[1] for p in graph.points], marker="D")
        for i, p in enumerate(graph.points):
            axes[2].annotate(i, (p[0], p[1]))
        axes[2].set_title(f"Solution {cr} crossings", fontsize=FONT_SIZE)

        plt.savefig(OUTPUT_PATH, bbox_inches='tight')


if __name__ == '__main__':
    os.system("mkdir -p out/solutions")
    with Pool(24) as p:
        p.map(plot, glob.glob("out/*.json"))

