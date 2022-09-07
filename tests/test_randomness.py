"""Tests for the randomness across Topics environments."""

import numpy as np
from reclab.environments import Topics


def _test_env_seed(env1, env2):
    env1.seed(0)
    env2.seed(0)
    env1.reset()
    env2.reset()

    assert env1._user_preferences.all() == env2._user_preferences.all()
    assert env1._item_topics.all() == env2._item_topics.all()

    for i in range(5):
        env1.step(np.array([[i]]))
        env2.step(np.array([[i]]))

    assert env1._get_dense_ratings().all() == env2._get_dense_ratings().all()
    assert env1._satiations.all() == env2._satiations.all()
    assert env1._sensitization_state.all() == env2._sensitization_state.all()
    assert env1._user_biases.all() == env2._user_biases.all()
    assert env1._item_biases.all() == env2._item_biases.all()


def test_envs():
    static_env1 = Topics(num_topics=2,
                         num_users=4,
                         num_items=10,
                         rating_frequency=0.25,
                         num_init_ratings=0,
                         noise=0.0)
    static_env2 = Topics(num_topics=2,
                         num_users=4,
                         num_items=10,
                         rating_frequency=0.25,
                         num_init_ratings=0,
                         noise=0.0)
    _test_env_seed(static_env1, static_env2)

    dynamic_env1 = Topics(num_topics=2,
                          num_users=4,
                          num_items=10,
                          rating_frequency=0.25,
                          num_init_ratings=0,
                          noise=0.0,
                          topic_change=0.1,
                          memory_length=5,
                          boredom_threshold=2,
                          boredom_penalty=1)
    dynamic_env2 = Topics(num_topics=2,
                          num_users=4,
                          num_items=10,
                          rating_frequency=0.25,
                          num_init_ratings=0,
                          noise=0.0,
                          topic_change=0.1,
                          memory_length=5,
                          boredom_threshold=2,
                          boredom_penalty=1)
    _test_env_seed(dynamic_env1, dynamic_env2)

    shift_env1 = Topics(num_topics=2,
                        num_users=4,
                        num_items=10,
                        rating_frequency=0.25,
                        num_init_ratings=0,
                        noise=0.0,
                        user_dist_choice='uniform',
                        shift_steps=2,
                        shift_frequency=1,
                        shift_weight=0.5,
                        user_bias_type='normal')
    shift_env2 = Topics(num_topics=2,
                        num_users=4,
                        num_items=10,
                        rating_frequency=0.25,
                        num_init_ratings=0,
                        noise=0.0,
                        user_dist_choice='uniform',
                        shift_steps=2,
                        shift_frequency=1,
                        shift_weight=0.5,
                        user_bias_type='normal')
    _test_env_seed(shift_env1, shift_env2)

    satiation_env1 = Topics(num_topics=2,
                            num_users=4,
                            num_items=10,
                            rating_frequency=0.25,
                            num_init_ratings=0,
                            noise=0.0,
                            satiation_factor=3,
                            satiation_decay=0.5,
                            satiation_noise=0.1)
    satiation_env2 = Topics(num_topics=2,
                            num_users=4,
                            num_items=10,
                            rating_frequency=0.25,
                            num_init_ratings=0,
                            noise=0.0,
                            satiation_factor=3,
                            satiation_decay=0.5,
                            satiation_noise=0.1)
    _test_env_seed(satiation_env1, satiation_env2)

    sensitization_env1 = Topics(num_topics=2,
                                num_users=4,
                                num_items=10,
                                rating_frequency=0.25,
                                num_init_ratings=0,
                                noise=0.0,
                                satiation_factor=3,
                                satiation_decay=(0.1, 0.5),
                                satiation_noise=0.1,
                                switch_probability=(0.05, 0.2))
    sensitization_env2 = Topics(num_topics=2,
                                num_users=4,
                                num_items=10,
                                rating_frequency=0.25,
                                num_init_ratings=0,
                                noise=0.0,
                                satiation_factor=3,
                                satiation_decay=(0.1, 0.5),
                                satiation_noise=0.1,
                                switch_probability=(0.05, 0.2))
    _test_env_seed(sensitization_env1, sensitization_env2)
