"""Contains the implementation for the TopicsGraph environment.

This environment inherits from the Topics environment, adding an underlying graph structure
representing connections between users. The opinion formation dynamics follow DeGroot's repeated averaging 
with biased assimilation, as described in "Biased Assimilation, Homophily, and the Dynamics of Polarization."

https://www.pnas.org/doi/pdf/10.1073/pnas.1217220110 
"""

import numpy as np
import networkx as nx
from .topics import Topics
import matplotlib.pyplot as plt


class TopicsGraph(Topics):
    """
    An environment where items belong to topics and users have different preferences for each topic.
    The users are connected by an underlying graph. The connections determine how strongly users
    are influenced by their neighbors when rating an item.

    The bias parameter determines the degree to which a user assimilates with their neighbors.
    Bias of 0 leads to consensus following DeGroot's repeated averaging.
    Bias >= 1 may lead to polarization.

    user_graph : networkx graph
        The network of users. Nodes represent users and edges represent connections between users.
        The number of nodes must equal the number of users, and node labels must correspond
        to each user_id. If unspecified, will be initialized as a barabasi albert graph. 
        Edge weights belong to the field "influence" and represent the cosine similarity 
        of the users' preference vectors across all topics.

    m: int
        Represents the number of edges to attach from a new node to existing nodes in a
        barabsi albert graph. If unspecified, will be 1/3 of the number of users.

    bias: float, list, or array
        The degree of bias with which a user assimilates to the opinions of their neighbors.
        If zero, opinion formation will follow DeGroot's repeated averaging opinion formation
        process. Must be >= 0.

    """

    def __init__(self, user_graph=None, m=None, bias=0.0, *args, **kwargs):
        """Create a TopicsGraph environment."""
        super().__init__(*args, **kwargs)
        self._user_preferences = None
        self._item_topics = None
        self._satiations = None
        self._sensitization_state = None
        self._user_biases = None
        self._item_biases = None
        self._offset = None
        self._user_graph = user_graph
        self._m = m
        self._bias = bias

    @property
    def name(self):
        return 'topics-graph'

    def _display_graph(self):
        nx.draw(self._user_graph, with_labels=True)
        plt.show()

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Sets edge weights for the user graph by instantiating an "influence" field of each edge.
    # Represents the influence/strength of the relationship between the corresponding nodes (users).
    def _update_edges(self):
        for i, j in self._user_graph.edges():
            opinions_i = self._user_preferences[i]
            opinions_j = self._user_preferences[j]
            # influence field of edge represents cosine similarity of the
            # neighbors' preferences across all topics
            self._user_graph[i][j]["influence"] = self._cosine_similarity(
                opinions_i, opinions_j)

    def _rate_items(self, user_id, item_ids):
        item_id = [item_ids[0]]
        topic = self._item_topics[item_id]

        # Update user's preference for topic based on neighbors' opinions of that topic
        neighbors = nx.all_neighbors(self._user_graph, user_id)
        weighted_degree = 0
        weighted_sum = 0
        for n in neighbors:
            influence = self._user_graph[user_id][n]["influence"]
            weighted_degree += influence
            weighted_sum += influence * \
                (self._user_preferences[n, topic] - 0.5) / 5

        curr_opinion = (self._user_preferences[user_id, topic] - 0.5) / 5
        new_opinion = (curr_opinion + pow(curr_opinion, self._bias[user_id]) * weighted_sum) / (1 + pow(
            curr_opinion, self._bias[user_id]) * weighted_sum + pow(1 - curr_opinion, self._bias[user_id]) * (weighted_degree - weighted_sum))

        self._user_preferences[user_id, topic] = 5 * new_opinion + 0.5

        return super()._rate_items(user_id, item_ids)

    def _reset_state(self):
        super()._reset_state()
        # Initialize user graph if none passed
        if self._user_graph == None:
            # If m not specified, will be 1/3 of num_users
            if self._m == None:
                self._m = self._num_users // 3

            # Default graph is undirected barabasi albert, connections based on preferential attachment
            self._user_graph = nx.barabasi_albert_graph(
                self._num_users, self._m, seed=self._init_random)

        assert len(self._user_graph.nodes()) == self._num_users, "wrong number nodes: expected " + \
            str(self._num_users) + ", got: " + \
            str(len(self._user_graph.nodes()))

        # Set the edge weights. Only done once during the simulation
        self._update_edges()

        if type(self._bias) is float or type(self._bias) is int:
            self._bias = [self._bias] * self._num_users
        assert len(
            self._bias) == self._num_users, "incorrect number of users assigned a bias"
