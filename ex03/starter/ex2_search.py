import collections
import numpy as np
from scipy.spatial import distance
import itertools


class PriorityQueue:
    ###
    # Exercise: implement PriorityQueue
    ###
    def __init__(self):
        self.queue = []
        self.priority = []
        self.index = 0

    def empty(self):
        """
        :return: True if the queue is empty, False otherwise.
        """
        if self.index == 0:
            return True
        else:
            return False

    def add(self, item, priority=0):
        """
        Add item to the queue
        :param item: any object
        :param priority: int
        """
        self.queue = self.queue + [item]
        self.priority = self.priority + [priority]
        self.index = self.index + 1

    def pop(self):
        """
        Get the item with the minimal priority and remove it from the queue.
        :return: item with the minimal priority
        """
        if self.empty():
            return []
        ix_prior = np.argmin(self.priority)
        pop_item = self.queue[ix_prior]
        self.index = self.index - 1
        self.priority.pop(ix_prior)
        self.queue.pop(ix_prior)

        return pop_item

    def is_in(self, c):
        # checks wheter a candidate c is part of the queue
        return c in self.queue



def heuristic(node_a, node_b, norm='euclidean'):
    """
    Heuristic
    :param node_a: pair, (x_a, y_a)
    :param node_b: pair, (x_b, y_b)
    :return: estimated distance between node_a and node_b
    """
    if norm == 'euclidean':
        dist = distance.euclidean(node_a, node_b)
    else:
        dist = np.nan

    return dist


def a_star_search(graph, start, goal):
    """

    :param graph: SquareGrid, defines the graph where we build a route.
    :param start: pair, start node coordinates (x, y)
    :param goal: pair, goal node coordinates(x, y)
    :return:
        came_from: dict, with keys - coordinates of the nodes. came_from[X] is coordinates of
            the node from which the node X was reached.
            This dict will be used to restore final path.
        cost_so_far: dict,
    """
    came_from = dict()
    cost_so_far = dict()

    # instance of queue
    open_set = PriorityQueue()
    closed_set = PriorityQueue()
    open_set.add(start, heuristic(start, goal))

    came_from[start] = None
    cost_so_far[start] = 0

    while not open_set.empty():
        current_node = open_set.pop()

        if current_node == goal:
            return came_from, cost_so_far # reconstruct_path(came_from, start, current_node)

        closed_set.add(current_node)

        for n in graph.neighbors(current_node):
            if closed_set.is_in(n):
                continue

            if not open_set.is_in(n):
                open_set.add(n, heuristic(n, goal))

            tentative_g = cost_so_far[current_node] + graph.cost(current_node, n)
            if tentative_g >= cost_so_far:
                continue

            came_from[n] = current_node
            cost_so_far[n] = tentative_g

            open_set.add(n, cost_so_far[n] + heuristic(n, goal))

    assert True == False, 'Failure'


def reconstruct_path(came_from, start, goal):
    """
    Reconstruct path using came_from dict
    :param came_from: ict, with keys - coordinates of the nodes. came_from[X] is coordinates of
            the node from which the node X was reached.
    :param start: pair, start node coordinates (x, y)
    :param goal: pair, goal node coordinates(x, y)
    :return: path: list, contains coordinates of nodes in the path

    """
    current = goal
    path = []
    if goal not in came_from:
        return path
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start)
    path.reverse()
    return path

if __name__ == '__main__':
    # q = PriorityQueue()
    # q.add(3, 5)
    # q.add(3, 1)
    # q.add(2, 3)
    #
    # q.pop()
    print(heuristic([5,5],[7,7]))