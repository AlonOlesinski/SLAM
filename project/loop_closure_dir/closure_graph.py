import dijkstar
import numpy as np


class ClosureGraph:
    """
    A graph that represents the closure of the loop closure problem.
    """
    def __init__(self, c0: int):
        self.graph = dijkstar.Graph()
        self.graph.add_node(c0)
        self.edge_dict = {}

    def add_node(self, c: int):
        """
        add a node to the graph
        :param c: the id of the node
        """
        self.graph.add_node(c)

    def add_edge(self, c1: int, c2: int, weight: float, cov_matrix: np.ndarray):
        """
        add an edge to the graph
        :param c1: the id of the first node
        :param c2: the id of the second node
        :param weight: the weight of the edge
        :param cov_matrix: the covariance matrix of the edge
        """
        self.graph.add_edge(c1, c2, weight)
        self.edge_dict[(c1, c2)] = cov_matrix

    def shortest_path(self, ci: int, cn: int):
        """
        find the shortest path from ci to cn
        :param ci: the id of the source node
        :param cn: the id of the target node
        :return: the shortest path
        """
        return dijkstar.find_path(self.graph, ci, cn)

    def get_sum_cov(self, ci: int, cn: int):
        """
        get the sum of the covariance matrices of the edges on the shortest path from ci to cn
        :param ci: the id of the source node
        :param cn: the id of the target node
        :return: the sum of covariance matrices
        """
        shortest_path_info = self.shortest_path(ci, cn)
        prev_node_idx = ci
        sum_cov = np.zeros((6,6))
        for i in range(1, len(shortest_path_info.nodes)):
            cur_node_idx = shortest_path_info.nodes[i]
            sum_cov += self.edge_dict[(prev_node_idx, cur_node_idx)]
            prev_node_idx = cur_node_idx
        return sum_cov
