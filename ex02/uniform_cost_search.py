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
    edges = [(1, 2, {'weight': 71}), (2, 3, {'weight': 75}), (3, 4, {'weight': 118}), (4, 5, {'weight': 111}),
                 (5, 6, {'weight': 70}), (6, 7, {'weight': 75}), (7, 8, {'weight': 120}), (8, 9, {'weight': 138}),
                 (9, 10, {'weight': 101}), (10, 11, {'weight': 90}), (10, 12, {'weight': 85}), (12, 13, {'weight': 98}),
                 (13, 14, {'weight': 86}), (12, 15, {'weight': 142}), (15, 16, {'weight': 92}),
                 (16, 17, {'weight': 87}),
                 (10, 18, {'weight': 211}), (18, 19, {'weight': 99}), (19, 1, {'weight': 151}),
                 (3, 19, {'weight': 140}),
                 (19, 20, {'weight': 80}), (18, 10, {'weight': 211}), (20, 8, {'weight': 146}), (20, 9, {'weight': 97})]

    nodes = [(1, {'name': 'Oradea', 'sld': 380}), (2, {'name': 'Zerind', 'sld': 374}), (3, {'name': 'Arad', 'sld': 366}),
             (4, {'name': 'Timisoara', 'sld': 329}), (5, {'name': 'Lugoj', 'sld': 244}), (6, {'name': 'Mehadia', 'sld': 241}),
             (7, {'name': 'Drobeta', 'sld': 242}), (8, {'name': 'Craiova', 'sld': 160}), (9, {'name': 'Pitesti', 'sld': 100}),
             (10, {'name': 'Buchareset', 'sld': 0}), (11, {'name': 'Giurgiu', 'sld': 77}), (12, {'name': 'Urziceni', 'sld': 80}),
             (13, {'name': 'Hirsova', 'sld': 151}), (14, {'name': 'Eforie', 'sld': 161}), (15, {'name': 'Vaslui', 'sld': 199}),
             (16, {'name': 'Iasi', 'sld': 226}), (17, {'name': 'Neamt', 'sld': 234}), (18, {'name': 'Fagaras', 'sld': 176}),
             (19, {'name': 'Sibiu', 'sld': 253}), (20, {'name': 'Rimnicu Vilcea', 'sld': 193})]

    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    return G


def plot_graph(graph, highlighted_nodes=[]):
    node_colors = ["green" if n in highlighted_nodes else "red" for n in graph.nodes()]
    labels = {k: G.node[k]['name'] for k in highlighted_nodes}
    pos = nx.spring_layout(G)

    nx.draw_networkx(graph, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels)
    # plt.show()




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
                    # print("Current Node: " + graph.node[i]['name'] + "cost: " + str(cost_to_i) + '    path: ' +
                    #       str([graph.node[j]['name'] for j in (path + [i])]))

def greedy_best_first_search(graph, root, goal):
    frontier = PriorityQueue()
    # sld to goal
    frontier.put((graph.node[root]['sld'], root, [root]))
    # explored = set()

    while True:
        try:
            if frontier.empty():
                raise Exception('Empty Frontier')
        except NameError:
            print('Exception!')
            raise
        sld, node, path = frontier.get()
        if node == goal:
            return path

        for i in graph.neighbors(node):
            frontier.put((graph.node[i]['sld'], i, path + [i]))
            # print("Current Node: " + graph.node[i]['name'] + "sld: " + str(sld) + '    path: ' +
            #       str([graph.node[j]['name'] for j in (path + [i])]))

def is_best_path(best_path, test_path):
    if test_path == best_path:
        return True
    else:
        print("Best path: " + str(best_path) + " Actual Path: " + str(test_path))
        return False

def test_equality(graph, end_node):
    for start_node in graph:
        _, path_u = uniform_cost_search(G, start_node, end_node)
        path = greedy_best_first_search(G, start_node, end_node)
        print("Current node: " + graph.node[start_node]['name'])
        is_best_path(path_u, path)

if __name__ == '__main__':
    G = create_romania_graph()
    start_node = 4  # 'Timisoara'
    end_node = 10   # 'Bucharest'

    total_cost, path_u = uniform_cost_search(G, start_node, end_node)
    path = greedy_best_first_search(G, start_node, end_node)
    print("path: " + str([G.node[i]['name'] for i in path]))
    plot_graph(G, path)
    plt.savefig('figures/graph.pdf')
    plt.close()

    test_equality(G, end_node)