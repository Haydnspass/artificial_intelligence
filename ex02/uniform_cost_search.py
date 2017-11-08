"""
    Artificial Intelligence
    Prof. Bjoern Ommer
    WS17/18
    Ex02
"""
from queue import PriorityQueue
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def create_romania_graph():
    G = nx.Graph()
    edge_list = [(1, 2, {'weight': 71}), (2, 3, {'weight': 75}), (3, 4, {'weight': 118}), (4, 5, {'weight': 111}),
                 (5, 6, {'weight': 70}), (6, 7, {'weight': 75}), (7, 8, {'weight': 120}), (8, 9, {'weight': 138}),
                 (9, 10, {'weight': 101}), (10, 11, {'weight': 90}), (10, 12, {'weight': 85}), (12, 13, {'weight': 98}),
                 (13, 14, {'weight': 86}), (12, 15, {'weight': 142}), (15, 16, {'weight': 92}), (16, 17, {'weight': 87}),
                 (10, 18, {'weight': 211}), (18, 19, {'weight': 99}), (19, 1, {'weight': 151}), (3, 19, {'weight': 140}),
                 (19, 20, {'weight': 80}), (18, 10, {'weight': 211}), (20, 8, {'weight': 146}), (20, 9, {'weight': 97})]
    G.add_edges_from(edge_list)

    return G


def plot_graph(graph):
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def uniform_cost_search(graph, root, goal):
    frontier = PriorityQueue()
    frontier.put((0, root))
    explored = set()

    found_optimal_path = False
    while not found_optimal_path:
        cost, node = frontier.get()
        # try:
        #     if frontier.empty():
        #         raise Exception('Empty Frontier')
        # except NameError:
        #     print('Exception!')
        #     raise
        if node not in explored:
            explored.add(node)

            if node == goal:
                return
            for i in graph.neighbors(node):
                if i not in explored:
                    cost_to_i = cost + graph.get_edge_data(node, i)['weight']
                    frontier.put((cost_to_i, i))

    # return path


if __name__ == '__main__':
    G = create_romania_graph()
    start_node = 4  # 'Timisoara'
    end_node = 10  # 'Bucharest'

    uniform_cost_search(G, start_node, end_node)
    # plot_graph(G)