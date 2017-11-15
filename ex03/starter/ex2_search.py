import collections


class PriorityQueue:
    ###
    # Exercise: implement PriorityQueue
    ###
    def __init__(self):
        pass

    def empty(self):
        """
        :return: True if the queue is empty, False otherwise.
        """
        pass

    def add(self, item, priority):
        """
        Add item to the queue
        :param item: any object
        :param priority: int
        """
        pass

    def pop(self):
        """
        Get the item with the minimal priority and remove it from the queue.
        :return: item with the minimal priority
        """
        pass


def heuristic(node_a, node_b):
    """
    Heuristic
    :param node_a: pair, (x_a, y_a)
    :param node_b: pair, (x_b, y_b)
    :return: estimated distance between node_a and node_b
    """
    ###
    # Exercise: implement a heuristic for A* search
    ###
    return 0


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
    ###
    # Exercise: implement A* search
    ###
    return came_from, cost_so_far


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
