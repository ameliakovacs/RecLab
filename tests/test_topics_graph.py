"""Tests for the TopicsGraph environment."""

import numpy as np
import networkx as nx
from reclab.environments import TopicsGraph


def test_graph_creation():
    env = TopicsGraph(num_topics=2,
                      num_users=3,
                      num_items=2,
                      rating_frequency=1.0,
                      num_init_ratings=0,
                      noise=0.0,
                      bias=0.0)
    env.reset()

    assert len(env._user_graph.nodes()) == 3
    assert len(env._bias) == 3
    assert env._m == 1


def test_degroot():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    env = TopicsGraph(num_topics=2,
                      num_users=2,
                      num_items=2,
                      rating_frequency=0.5,
                      num_init_ratings=0,
                      noise=0.0,
                      user_graph=G,
                      bias=0.0)
    env.seed(0)
    env.reset()
    topic = env._item_topics[0]
    # seed 0 makes user 1 the first online
    old_pref0, old_pref1 = np.copy(
        env._user_preferences[0]), np.copy(env._user_preferences[1])
    influence = env._user_graph[0][1]["influence"]
    # recommend item 0 to user 1
    env.step(np.array([[0]]))
    new_pref0, new_pref1 = np.copy(
        env._user_preferences[0]), np.copy(env._user_preferences[1])
    # DeGroot repeated averaging
    assert round(new_pref1[topic], 10) == round(
        (old_pref1[topic] + influence * env._user_preferences[0][topic]) / (1 + influence), 10)
    assert new_pref1[:topic].all() == old_pref1[:topic].all(
    ) and new_pref1[topic + 1:].all() == old_pref1[topic + 1:].all()
    assert new_pref0.all() == old_pref0.all()


def test_bias():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    env = TopicsGraph(num_topics=2,
                      num_users=2,
                      num_items=2,
                      rating_frequency=0.5,
                      num_init_ratings=0,
                      noise=0.0,
                      user_graph=G,
                      bias=1.0)
    env.seed(0)
    env.reset()
    topic = env._item_topics[0]
    # seed 0 makes user 1 the first online
    old_pref0, old_pref1 = np.copy(
        env._user_preferences[0]), np.copy(env._user_preferences[1])
    # recommend item 0 to user 1
    env.step(np.array([[0]]))
    new_pref0, new_pref1 = np.copy(
        env._user_preferences[0]), np.copy(env._user_preferences[1])
    assert new_pref1[topic] != old_pref1[topic]
    assert new_pref1[:topic].all() == old_pref1[:topic].all(
    ) and new_pref1[topic + 1:].all() == old_pref1[topic + 1:].all()
    assert new_pref0.all() == old_pref0.all()
