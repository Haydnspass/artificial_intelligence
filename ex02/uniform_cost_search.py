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

    
    G.node[3]['name'] = 'Arad'
    G.node[4]['name'] = 'Timisoara'
    G.node[9]['name'] = 'Pitesti'
    G.node[10]['name'] = 'Buchareset'
    G.node[19]['name'] = 'Sibiu'
    G.node[20]['name'] = 'Rimnicu Vilcea'



    return G


def plot_graph(graph):
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def uniform_cost_search(graph, root, goal):
    frontier = PriorityQueue()
    frontier.put((0, root, [root]))
    explored = set()

    while True:
        try:
            if frontier.empty():
                raise Exception('Empty Frontier')
        except NameError:
            print('Exception!')
            raise
        cost, node, path = frontier.get()

        if node not in explored:
            explored.add(node)

            if node == goal:
                return cost, path
            for i in graph.neighbors(node):
                if i not in explored:
                    cost_to_i = cost + graph.get_edge_data(node, i)['weight']
                    frontier.put((cost_to_i, i, path + [i]))


if __name__ == '__main__':
    G = create_romania_graph()
    start_node = 4  # 'Timisoara'
    end_node = 10  # 'Bucharest'

    total_cost, path = uniform_cost_search(G, start_node, end_node)
    print("Total pathlength was: " + str(total_cost) + "km, using the path: " + str([G.node[i]['name'] for i in path]))
    # plot_graph(G)