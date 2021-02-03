import math
import sys

import numpy as np

sys.path.append('../')
sys.path.append('../../')
from env_defaults import *
from experiment import get_env_dataset, sample_ratings
from tuner import ModelTuner
from reclab.environments import Topics
from reclab.recommenders import LibFM

def test_offline(user_dist_choice, initial_sampling, use_mse, sample=False, low=False):
    # Environment setup
    environment_name = TOPICS_STATIC['name']
    env = Topics(**TOPICS_STATIC['params'], **TOPICS_STATIC['optional_params'],
                 user_dist_choice=user_dist_choice, initial_sampling=initial_sampling)
    env.seed(0)

    # Recommender setup
    recommender_class = LibFM


    # ====Step 5====
    if sample:
        starting_data = sample_ratings(env, low_ratings=low)
    else:
        starting_data = get_env_dataset(env)


    # ====Step 6====
    # Recommender tuning setup
    n_fold = 5
    num_users, num_items = get_num_users_items(TOPICS_STATIC)
    default_params = dict(num_user_features=0,
                          num_item_features=0,
                          num_rating_features=0,
                          max_num_users=num_users,
                          max_num_items=num_items,
                          method='sgd')
    tuner = ModelTuner(starting_data,
                       default_params,
                       recommender_class,
                       n_fold=n_fold,
                       verbose=True,
                       use_mse=use_mse)

    # Tune the performance independent hyperparameters.
    bias_reg = [0.07]
    init_stdev = [1.0]
    learning_rate = [0.01]
    num_iter = [200]
    num_two_way_factors = [20]
    reg = [0.07]

    results = tuner.evaluate_grid(
        bias_reg=bias_reg,
        init_stdev=init_stdev,
        learning_rate=learning_rate,
        num_iter=num_iter,
        num_two_way_factors=num_two_way_factors,
        reg=reg)

test_offline('uniform', 'uniform', False, True, True)
test_offline('uniform', 'uniform', True, True, True)
print('=========POWERLAW============')
test_offline('powerlaw', 'powerlaw', False)
test_offline('powerlaw', 'powerlaw', True)
print('=========UNIFORM============')
test_offline('uniform', 'uniform', False)
test_offline('uniform', 'uniform', True)
print('=========POWERLAW RATINGS HIGH============')
test_offline('uniform', 'uniform', False, True)
test_offline('uniform', 'uniform', True, True)
print('=========POWERLAW RATINGS LOW============')
test_offline('uniform', 'uniform', False, True, True)
test_offline('uniform', 'uniform', True, True, True)
